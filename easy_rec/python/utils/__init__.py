class conditional(object):
  """Wrap another context manager and enter it only if condition is true."""

  def __init__(self, condition, contextmanager):
    self.condition = condition
    self.contextmanager = contextmanager

  def __enter__(self):
    """Conditionally enter a context manager."""
    if self.condition:
      return self.contextmanager.__enter__()

  def __exit__(self, *args):
    if self.condition:
      return self.contextmanager.__exit__(*args)
