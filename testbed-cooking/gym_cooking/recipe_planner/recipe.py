from gym_cooking.utils.core import *


class Recipe:
    def __init__(self, name, length, bonus):
        self.name = name
        self.length = length
        self.bonus = bonus
        self.contents = []

    def __str__(self):
        return self.name

    def add_ingredient(self, item):
        self.contents.append(item)

    def add_goal(self):
        self.contents = sorted(self.contents, key = lambda x: x.name)   # list of Food objects
        self.contents_names = [c.full_name for c in self.contents]   # list of strings
        self.full_name = '-'.join(sorted(self.contents_names))   # string
        self.full_plate_name = '-'.join(sorted(self.contents_names + ['Plate']))   # string
        self.final_task = Object(location=None, contents=self.contents + [Plate()])

class SimpleTomato(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'Tomato', 20, 10)
        self.add_ingredient(Tomato(state_index=2))
        self.add_goal()

class SimpleLettuce(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'Lettuce', 20, 10)
        self.add_ingredient(Lettuce(state_index=2))
        self.add_goal()

class SimpleOnion(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'Onion', 20, 10)
        self.add_ingredient(Onion(state_index=2))

class TomatoLettuceSalad(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'TomatoLettuceSalad', 30, 15)
        self.add_ingredient(Tomato(state_index=2))
        self.add_ingredient(Lettuce(state_index=2))
        self.add_goal()

class OnionTomatoSalad(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'OnionTomatoSalad', 30, 15)
        self.add_ingredient(Onion(state_index=2))
        self.add_ingredient(Tomato(state_index=2))
        self.add_goal()

class OnionLettuceSalad(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'OnionLettuceSalad', 30, 15)
        self.add_ingredient(Onion(state_index=2))
        self.add_ingredient(Lettuce(state_index=2))
        self.add_goal()

class FullSalad(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'FullSalad', 40, 20)
        self.add_ingredient(Tomato(state_index=2))
        self.add_ingredient(Lettuce(state_index=2))
        self.add_ingredient(Onion(state_index=2))
        self.add_goal()

class OnionSoup(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'OnionSoup', 50, 10)
        self.add_ingredient(Onion(state_index=4))

class TomatoSoup(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'TomatoSoup', 50, 10)
        self.add_ingredient(Tomato(state_index=4))
        
class LettuceSoup(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'LettuceSoup', 50, 10)
        self.add_ingredient(Lettuce(state_index=4))

class TomatoLettuceSoup(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'TomatoLettuceSoup', 60, 15)
        self.add_ingredient(Tomato(state_index=4))
        self.add_ingredient(Lettuce(state_index=4))
        self.add_goal()

class OnionTomatoSoup(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'OnionTomatoSoup', 60, 15)
        self.add_ingredient(Onion(state_index=4))
        self.add_ingredient(Tomato(state_index=4))
        self.add_goal()

class OnionLettuceSoup(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'OnionLettuceSoup', 60, 15)
        self.add_ingredient(Onion(state_index=4))
        self.add_ingredient(Lettuce(state_index=4))
        self.add_goal()

class FullSoup(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'FullSoup', 70, 20)
        self.add_ingredient(Tomato(state_index=4))
        self.add_ingredient(Lettuce(state_index=4))
        self.add_ingredient(Onion(state_index=4))
        self.add_goal()

