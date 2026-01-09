import copy

class UndoManager:
    def __init__(self):
        self.stack = []

    def save(self, images):
        self.stack.append(copy.deepcopy(images))

    def undo(self):
        if self.stack:
            return self.stack.pop()
        return None
