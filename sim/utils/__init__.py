import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# detr 내부의 'from util.misc import ...' 해결용
_DETR = os.path.join(_ROOT, 'detr')
if _DETR not in sys.path:
    sys.path.insert(0, _DETR)
