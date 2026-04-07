from abc import ABC, abstractmethod

# Component
class OrganizationMember(ABC):
    @abstractmethod
    def show_details(self, indent=0):
        pass

# Leaf
class Employee(OrganizationMember):
    def __init__(self, name, role):
        self.name = name
        self.role = role

    def show_details(self, indent=0):
        print(" " * indent + f"- {self.role}: {self.name}")


# Composite
class Manager(OrganizationMember):
    def __init__(self, name, role):
        self.name =  name
        self.role = role
        self.subordinates = []

    def add(self, member: OrganizationMember):
        self.subordinates.append(member)

    def remove(self, member: OrganizationMember):
        self.subordinates.remove(memeber)

    def show_details(self, indent=0):
        print(" " * indent + f"+ {self.role}: {self.name}")
        for member in self.subordinates:
            member.show_details(indent + 2)

if __name__ == "__main__":
    # Create Employees
    dev1 = Employee("Alice", "Developer")
    dev2 = Employee("Bob", "Developer")
    designer = Employee("Charlie", "Designer")
    qa = Employee("Diana", "QA Engineer")

    # Create managers
    team_lead = Manager("Eve", "Team Lead")
    project_manager = Manager("Frank", "Project Manager")

    # Build hierarchy
    team_lead.add(dev1)
    team_lead.add(dev2)
    team_lead.add(designer)
    project_manager.add(team_lead)
    project_manager.add(qa)

    # show full org chart
    project_manager.show_details()

