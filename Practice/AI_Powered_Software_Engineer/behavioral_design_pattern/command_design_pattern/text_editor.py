from abc import ABC, abstractmethod

# Commnad Interface
class Command(ABC):
    @abstractmethod
    def execute(self):
        pass

    @abstractmethod
    def undo(self):
        pass

# Receiver
class TextEditor:
    def __init__(self):
        self.content = ""

    def write(self, text):
        self.content += text
        return f"Wrote: '{text}' | Current content: '{self.content}'"

    def delete(self, count):
        deleted = self.content[-count:]
        self.content = self.content[:-count]
        return f"Deleted: '{deleted}' | Current content: '{self.content}'"

# Concrete Commands
class WriteCommand(Command):
    def __init__(self, editor: TextEditor, text: str):
        self.editor = editor
        self.text = text

    def execute(self):
        return self.editor.write(self.text)

    def undo(self):
        return self.editor.delete(len(self.text))

class DeleteCommand(Command):
    def __init__(self, editor: TextEditor, count: int):
        self.editor = editor
        self.count = count
        self.deleted_text = ""

    def execute(self):
        self.deleted_text = self.editor.content[-self.count:]
        return self.editor.delete(self.count)

    def undo(self):
        return self.editor.write(self.deleted_text)

# Invoker
class CommandManager:
    def __init__(self):
        self._history = []

    def execute_command(self, command: Command):
        result = command.execute()
        self._history.append(command)
        print(result)

    def undo_last(self):
        if self._history:
            command = self._history.pop()
            result = command.undo()
            print(f"Undo: {result}")

if __name__ == "__main__":
    editor = TextEditor()
    manager = CommandManager()


    # Perform commands
    manager.execute_command(WriteCommand(editor, "Hello"))
    manager.execute_command(WriteCommand(editor, "World"))
    manager.execute_command(DeleteCommand(editor, 5)) 

    # Undo last two actions
    manager.undo_last()
    manager.undo_last()          