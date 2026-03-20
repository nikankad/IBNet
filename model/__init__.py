__all__ = ["QuartzNetBxR"]


def __getattr__(name):
	if name == "QuartzNetBxR":
		from .model import QuartzNetBxR
		return QuartzNetBxR
	raise AttributeError(f"module 'model' has no attribute {name!r}")