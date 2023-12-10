#
# Module implementing queues
#
# multiprocessing/queues.py
#
# Copyright (c) 2006-2008, R Oudkerk
# Licensed to PSF under a Contributor Agreement.
#

import collections
import errno
import logging
import os
import sys
import threading
import time
import weakref
from multiprocessing import connection
from multiprocessing.util import Finalize
from multiprocessing.util import is_exiting
from multiprocessing.util import register_after_fork
from queue import Empty
from queue import Full

import six

try:
  from multiprocessing import context
except ImportError:
  context = None
  pass

if context is not None:
  _ForkingPickler = context.reduction.ForkingPickler
else:
  _ForkingPickler = None

#
# Queue type using a pipe, buffer and thread
#


class Queue(object):

  _sentinel = object()

  def __init__(self, ctx, maxsize=0, name=''):
    assert not six.PY2, 'python2 is not supported'
    if maxsize <= 0:
      # Can raise ImportError (see issues #3770 and #23400)
      from multiprocessing.synchronize import SEM_VALUE_MAX as maxsize
    self._maxsize = maxsize
    self._reader, self._writer = connection.Pipe(duplex=False)
    self._rlock = ctx.Lock()
    self._opid = os.getpid()
    if sys.platform == 'win32':
      self._wlock = None
    else:
      self._wlock = ctx.Lock()
    self._sem = ctx.BoundedSemaphore(maxsize)
    # For use by concurrent.futures
    self._ignore_epipe = False
    self._reset()
    self._name = name
    self._run = True

    if sys.platform != 'win32':
      register_after_fork(self, Queue._after_fork)

  def __getstate__(self):
    context.assert_spawning(self)
    return (self._ignore_epipe, self._maxsize, self._reader, self._writer,
            self._rlock, self._wlock, self._sem, self._opid, self._name,
            self._run)

  def __setstate__(self, state):
    (self._ignore_epipe, self._maxsize, self._reader, self._writer, self._rlock,
     self._wlock, self._sem, self._opid, self._name, self._run) = state
    self._reset()

  def _after_fork(self):
    logging.debug('Queue._after_fork()')
    self._reset(after_fork=True)

  def _reset(self, after_fork=False):
    if after_fork:
      self._notempty._at_fork_reinit()
    else:
      self._notempty = threading.Condition(threading.Lock())
    self._buffer = collections.deque()
    self._thread = None
    self._jointhread = None
    self._joincancelled = False
    self._closed = False
    self._close = None
    self._send_bytes = self._writer.send_bytes
    self._recv_bytes = self._reader.recv_bytes
    self._poll = self._reader.poll

  def put(self, obj, block=True, timeout=None):
    if self._closed:
      raise ValueError('Queue %s is closed' % self._name)
    if not self._sem.acquire(block, timeout):
      raise Full

    with self._notempty:
      if self._thread is None:
        self._start_thread()
      self._buffer.append(obj)
      self._notempty.notify()

  def get(self, block=True, timeout=None):
    if self._closed:
      raise ValueError('Queue %s is closed' % self._name)
    if block and timeout is None:
      with self._rlock:
        res = self._recv_bytes()
      self._sem.release()
    else:
      if block:
        deadline = time.monotonic() + timeout
      if not self._rlock.acquire(block, timeout):
        raise Empty
      try:
        if block:
          timeout = deadline - time.monotonic()
          if not self._poll(timeout):
            raise Empty
        elif not self._poll():
          raise Empty
        res = self._recv_bytes()
        self._sem.release()
      finally:
        self._rlock.release()
    # unserialize the data after having released the lock
    return _ForkingPickler.loads(res)

  def qsize(self):
    # Raises NotImplementedError on Mac OSX because of broken sem_getvalue()
    return self._maxsize - self._sem._semlock._get_value()

  def empty(self):
    return not self._poll()

  def full(self):
    return self._sem._semlock._is_zero()

  def get_nowait(self):
    return self.get(False)

  def put_nowait(self, obj):
    return self.put(obj, False)

  def close(self, wait_send_finish=True):
    self._closed = True
    close = self._close
    if not wait_send_finish and self._thread is not None and self._thread.is_alive(
    ):
      try:
        if self._reader is not None:
          self._reader.close()
      except Exception:
        pass
      self._run = False
      # clear queue
      # with self._rlock:
      #   while self._thread.is_alive() and self._poll(1):
      #     res = self._recv_bytes()
      #     logging.info('Queue[name=' + self._name + '] clear one elem')
      # logging.info('Queue[name=' + self._name + '] clear queue done')
    if close:
      self._close = None
      close()

  def join_thread(self):
    logging.debug('Queue.join_thread()')
    assert self._closed, 'Queue {0!r} not closed'.format(self)
    if self._jointhread:
      self._jointhread()

  def cancel_join_thread(self):
    logging.debug('Queue.cancel_join_thread()')
    self._joincancelled = True
    try:
      self._jointhread.cancel()
    except AttributeError:
      pass

  def _start_thread(self):
    logging.debug('Queue._start_thread()')

    # Start thread which transfers data from buffer to pipe
    self._buffer.clear()
    self._thread = threading.Thread(
        target=self._feed,
        args=(self._buffer, self._notempty, self._send_bytes, self._wlock,
              self._reader.close, self._writer.close, self._ignore_epipe,
              self._on_queue_feeder_error, self._sem),
        name='QueueFeederThread')
    self._thread.daemon = True

    logging.debug('doing self._thread.start()')
    self._thread.start()
    logging.debug('... done self._thread.start()')

    if not self._joincancelled:
      self._jointhread = Finalize(
          self._thread,
          Queue._finalize_join, [weakref.ref(self._thread)],
          exitpriority=-5)

    # Send sentinel to the thread queue object when garbage collected
    self._close = Finalize(
        self,
        Queue._finalize_close, [self._buffer, self._notempty],
        exitpriority=10)

  @staticmethod
  def _finalize_join(twr):
    logging.debug('joining queue thread')
    thread = twr()
    if thread is not None:
      thread.join()
      logging.debug('... queue thread joined')
    else:
      logging.debug('... queue thread already dead')

  @staticmethod
  def _finalize_close(buffer, notempty):
    logging.debug('telling queue thread to quit')
    with notempty:
      buffer.append(Queue._sentinel)
      notempty.notify()

  def _feed(self, buffer, notempty, send_bytes, writelock, reader_close,
            writer_close, ignore_epipe, onerror, queue_sem):
    logging.debug('starting thread to feed data to pipe')
    nacquire = notempty.acquire
    nrelease = notempty.release
    nwait = notempty.wait
    bpopleft = buffer.popleft
    sentinel = Queue._sentinel
    if sys.platform != 'win32':
      wacquire = writelock.acquire
      wrelease = writelock.release
    else:
      wacquire = None

    pid = os.getpid()
    name = self._name
    while self._run:
      try:
        nacquire()
        try:
          if not buffer:
            nwait()
        finally:
          nrelease()
        try:
          while self._run:
            obj = bpopleft()
            if obj is sentinel:
              # logging.info('Queue[' + self._name + '] feeder thread got sentinel -- exiting: ' + str(self._run))
              reader_close()
              writer_close()
              return

            # serialize the data before acquiring the lock
            obj = _ForkingPickler.dumps(obj)
            if wacquire is None:
              send_bytes(obj)
            else:
              wacquire()
              try:
                send_bytes(obj)
              finally:
                wrelease()
        except IndexError:
          pass
      except Exception as e:
        if ignore_epipe and getattr(e, 'errno', 0) == errno.EPIPE:
          logging.warning('Queue[' + name + '] exception: pid=' + str(pid) +
                          ' run=' + str(self._run) + ' e=' + str(e))
          return
        # Since this runs in a daemon thread the resources it uses
        # may be become unusable while the process is cleaning up.
        # We ignore errors which happen after the process has
        # started to cleanup.
        if is_exiting():
          logging.warning('Queue[' + name + '] thread error in exiting: pid=' +
                          str(pid) + ' run=' + str(self._run) + ' e=' + str(e))
          return
        else:
          # Since the object has not been sent in the queue, we need
          # to decrease the size of the queue. The error acts as
          # if the object had been silently removed from the queue
          # and this step is necessary to have a properly working
          # queue.
          queue_sem.release()
          onerror(e, obj)
    # logging.info('Queue[' + name +  '] send thread finish: pid=' + str(pid)
    #    +  ' run=' + str(self._run))

  @staticmethod
  def _on_queue_feeder_error(e, obj):
    """Private API hook called when feeding data in the background thread raises an exception.

    For overriding by concurrent.futures.
    """
    import traceback
    traceback.print_exc()
