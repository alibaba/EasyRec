import logging

from docutils import nodes
from docutils.transforms import Transform


class PostFixLink(Transform):
  default_priority = 780

  def __init__(self, document, startnode=None):
    super(PostFixLink, self).__init__(document, startnode)

  def apply(self, **kwargs):
    def _visit(node):
      if not node.children:
        return
      for child in node.children:
        if isinstance(child, nodes.Element):
          if 'refuri' in child.attributes and '.md#' in child.attributes['refuri']:
            src = child.attributes['refuri']
            dst = src.replace('.md#', '.html#')
            logging.info('[PostFixLink] replace %s to %s' % (src, dst))
            child.attributes['refuri'] = dst
        _visit(child)

    _visit(self.document)


def setup(app):
  app.add_post_transform(PostFixLink)

  return {
    'version': '0.1',
    'parallel_read_safe': True,
    'parallel_write_safe': True,
  }
