---
title: "OOPs for Data Scientists"
date: 2023-11-21T18:36:11-04:00
draft: false
author: "Kaushik Moudgalya"
showtoc: true
tags: ["Data Science", "OOPs", "Python"]
math: false
---

There are 4 principles of OOPs that everyone is expected to know. I used to be one of those people who thought that it was enough to know how to read and write OOPs code without actually knowing what those principles were as I thought they were more of a software engineering concept. But it makes sense to be aware of the common terms and verbalizations of the concepts within OOPs. 

Some Data Science and Machine Learning interviews will also test your OOPs knowledge. This post is an attempt to prepare you for those interviews. After all, failing to prepare is preparing to fail, right? (ᵔᴥᵔ)

[View this post in a Jupyter Notebook](https://github.com/Etrama/oops-for-data-scientists/blob/main/oops-for-data-scientists.ipynb)

Along the way we will also look at a couple of tricks and tips to help you code more easily. For now, we'll start with some an import:


```python
import random
```

First things first, the 4 principles of OOPs are:
- **A**bstraction
- **E**ncapsulation
- **I**nheritance
- **P**olymorphism  

If you need a mnemonic to remember them:  
    **AEIP** = **A**stronauts **E**xplore **I**nvisible **P**lanets.

## Classes

Let's make sure we have the basics of a class down. A class is a blueprint. The best example I could come up with for classes as a blueprint was to illustrate with a pokemon class:


```python
class Pokemon:
    pass
```

This class is pretty sad and does nothing. Let's rectify that. If you've ever played any pokemon based games, you know that each pokemon has its own Attack, Special Attack, Defense and Special Defense. Let's add those to our class.


```python
class Pokemon:
    def __init__(self):
        self.attack = 0
        self.special_attack = 0
        self.defense = 0
        self.special_defense = 0
```


```python
pikachu = Pokemon()
```

Throughout this post, we will refer to instances and objects and they mean one and the same thing, an instance / object of a class. "pikachu" is an instance or object of the Pokemon class.

## Know thy "self" 😉

Something that not everyone knows about the "self" keyword is that it doesn't have to be self at all. It can be anything, as long as we are consistent about it:


```python
class Pokemon_not_self:
    def __init__(not_self):
        not_self.attack = 10
        not_self.special_attack = 10
        not_self.defense = 10
        not_self.special_defense = 10
```


```python
example = Pokemon_not_self()
example.attack
```




    10



Though it is probably best to use the "self" keyword, as it is the convention and might confuse other people who are reading your code. Code is read more than it is written, so better follow the convention that helps readability. Let's make the Pokemon class a little more fun by randomly allowing created pokemons (pokemon objects) to be shiny. Then we'll jump into Inheritance.


```python
class Pokemon:
    type = 'normal'
    def __init__(self):
        self.attack = random.randint(0, 100)
        self.special_attack = random.randint(0, 100)
        self.defense = random.randint(0, 100)
        self.special_defense = random.randint(0, 100)
        self.hp = 100
        self.shiny = random.choices([True, False], weights=[0.0001, 0.9999], k=1)[0]
```

## Inheritance

Inheritance is the easiest to explain and understand, so let's start there. Let's say we want to create a class for each type (fire, water, grass, etc.) of pokemon. We can do that by inheriting from the Pokemon class:


```python
class PokemonFireType(Pokemon):
    type = 'fire'
```

So, let's say I create a Ratata, a normal typle pokemon and charmander, a fire type pokemon using the appropriate classes:


```python
ratata = Pokemon()
print(ratata.type)
for property in vars(ratata):
    print(property, ':', vars(ratata)[property])
```

    normal
    attack : 52
    special_attack : 25
    defense : 34
    special_defense : 68
    hp : 100
    shiny : False
    


```python
charmander = PokemonFireType()
print(charmander.type)
for property in vars(charmander):
    print(property, ':', vars(charmander)[property])
```

    fire
    attack : 87
    special_attack : 46
    defense : 32
    special_defense : 11
    hp : 100
    shiny : False
    

Even though we didn't define all the attributes for the PokemonTypeFire class, i.e. we didn't state that an instance of the PokemonFireType class would have attack, defense, .., shiny attributes, we see that it still does, because the PokemonTypeFire class inherited from the Pokemon class.

### Looping through instance attributes

Another little nuance, you can access the attribute values of an instance like so:


```python
class MyClass:
    def __init__(self):
        self.attribute1 = 1
        self.attribute2 = 2
        self.attribute3 = 3

object = MyClass()

for property in vars(object):
    print(property, ':', vars(object)[property])

```

    attribute1 : 1
    attribute2 : 2
    attribute3 : 3
    

I already feel more like a Software Enginner. Let's move on to the next principle.

## Polymorphism

Poly = many, morph = forms. Polymorphism is the ability of an object to take on many forms. Say we have the following scenario (changing the class names a bit for consistency):


```python
class PokemonNormalType:
    type = 'normal'
    def __init__(self):
        self.attack = random.randint(0, 100)
        self.special_attack = random.randint(0, 100)
        self.defense = random.randint(0, 100)
        self.special_defense = random.randint(0, 100)
        self.hp = 100
        self.shiny = random.choices([True, False], weights=[0.0001, 0.9999], k=1)[0]
        self.default_attack = "tackle"

class PokemonFireType(PokemonNormalType):
    type = 'fire'
    def __init__(self):
        self.default_attack = "ember"

class PokemonWaterType(PokemonNormalType):
    type = 'water'
    def __init__(self):
        self.default_attack = "water-gun"

class PokemonGrassType(PokemonNormalType):
    type = 'grass'
    def __init__(self):
        self.default_attack = "razor-leaf"
```

Given the above class definitions, we can retrieve their types and default attacks with a single function:


```python
ratata = PokemonNormalType()
charmander = PokemonFireType()
squirtle = PokemonWaterType()
bulbasaur = PokemonGrassType()
pokemons = [ratata, charmander, squirtle, bulbasaur]
```


```python
for pokemon in pokemons:
    print(f"Current pokemon type is {pokemon.type} and default attack is {pokemon.default_attack}")
```

    Current pokemon type is normal and default attack is tackle
    Current pokemon type is fire and default attack is ember
    Current pokemon type is water and default attack is water-gun
    Current pokemon type is grass and default attack is razor-leaf
    

The same object attribute / class property and even functions can take on different forms depending on the class that is calling it. This is polymorphism. Let's see a different example:


```python
class PokemonBasic:
    def __init__(self):
        self.height = random.randint(50, 200) # in cm
        self.weight = random.randint(10, 100) # in kgs
        self.hp = 100

class DigimonBasic:
    def __init__(self):
        self.height = random.randint(250, 350) # in cm
        self.weight = random.randint(200, 300) # in kgs
        self.hp = 99
```


```python
pikachu = PokemonBasic()
agumon = DigimonBasic()
all_the_mons = [pikachu, agumon]
for mon in all_the_mons:
    print(f"Height is {mon.height} cm and weight is {mon.weight} kgs")
```

    Height is 170 cm and weight is 86 kgs
    Height is 257 cm and weight is 222 kgs
    

There are two kinds of polymorphism when it comes to class methods:
- Method Overloading  
- Method Overriding<br>


### Method Overriding

Method Overriding is similar to what we already did before with instance attributes and class properties, just overriding methods instead. Override is the same as an overwrite, we write over the previously defined method with a newly defined method. Let's look at an example:


```python
class PokemonNormalType:
    def __init__(self):
        self.hp = 100
    def speak(self, sound):
        print(f"{sound[:4]}-{sound[4:]}!") # rata-ta!

class PokemonFireType(PokemonNormalType):
    def speak(self, sound):
        print(f"{sound[:4]}-{sound[:4]}!") # char-char!

class PokemonElectricType(PokemonNormalType):
    def speak(self, sound):
        print(f"{sound[:4]}-{sound[:2]}!") # pika-pi!
```


```python
ratata = PokemonNormalType()
charmander = PokemonFireType()
pikachu = PokemonElectricType()
pokemons = [ratata, charmander, pikachu]
sounds = ["ratata", "charmander", "pikachu"]
```


```python
for index, pokemon in enumerate(pokemons):
    pokemon.speak(sounds[index])
```

    rata-ta!
    char-char!
    pika-pi!
    

The method "speak" is overidden. The subclass PokemonFireType has its own speak method, which overrides the speak method of the parent PokemonNormalType class. The same thing happens vis-a-vis the PokemonElectricType and PokemonNormalType classes.

This overriding is evident when we look at the different outputs produced by the speak method run on the different instances.

Other languages (I'm looking at you, Java and C++), also have the concept of Method Overloading, where we can use the same method with a different number of arguments. For example:


```python
class PokeBag:
    def __init__(self):
        self.pokemons = []
    def add_pokemon(self, pokemon):
        self.pokemons.append(pokemon)
    def add_pokemon(self, pokemon1, pokemon2):
        self.pokemons.append(pokemon1)
        self.pokemons.append(pokemon2)
```

We could define a PokeBag class with the appropriate syntax in Java and it would work. But in Python, the latest method will override the previous declaration of that method. So, when we try to use the previous declaration of that method within Python, it will error out :


```python
bag = PokeBag()
try:
    bag.add_pokemon("ratata")
except Exception as e:
    print(f"Exception {e}")
```

    Exception PokeBag.add_pokemon() missing 1 required positional argument: 'pokemon2'
    

However, if we use the newly defined method, it will work:


```python
bag.add_pokemon("ratata", "charmander")
```

This is because Python is a dynamically typed language, i.e. we do not have to define the variable types hence they will be determined __dynamically__ when the program is run. This is in contrast to statically typed languages like Java and C++, where we have to define the variable types before we can use them. If python were statically typed, we would have to define the variable types for the methods in the class definition.

While we can define variable types in Python, it is meant to ease programming. The type hints are NOT enforced. As a best practice, you could use type hints to indicate variable types all the same, but make sure to remember that they are hints and not variable type assignments. Let's look at the PokeBag example with type hints:


```python
# PokeBag class with type hints
class PokeBag:
    def __init__(self):
        self.pokemons = []
    def add_pokemon(self, pokemon: str):
        self.pokemons.append(pokemon)
    def add_pokemon(self, pokemon1: str, pokemon2: str):
        self.pokemons.append(pokemon1)
        self.pokemons.append(pokemon2)
```

As the developer, I know that I need to pass pokemons as strings:


```python
bag = PokeBag()
bag.add_pokemon("ratata", "charmander")
bag.pokemons
```




    ['ratata', 'charmander']



But notice that the typing is not enforced:


```python
bag = PokeBag()
bag.add_pokemon("ratata", 2)
bag.pokemons
```




    ['ratata', 2]



Dynamic typing is one of the reasons why Python is slower than Java and C++, as there is more work to do when the program is run. With all the drama above, it may seem like method overloading is a pipe dream in Python.

## *args workaround

However, there is a way to achieve method overloading in Python. But first, if we just wish to call the method with an unknown number of arguments, we can just use the *args syntax :


```python
class PokeBag:
    def __init__(self):
        self.pokemons = []
    def add_pokemon(self, *pokemons):
        for pokemon in pokemons:
            self.pokemons.append(pokemon)
```


```python
bag = PokeBag()
bag.add_pokemon("ratata")
bag.pokemons
```




    ['ratata']




```python
bag = PokeBag()
bag.add_pokemon("pikachu")
bag.add_pokemon("ratata", "charmander")
bag.add_pokemon("squirtle", "bulbasaur", "metapod")
bag.pokemons
```




    ['pikachu', 'ratata', 'charmander', 'squirtle', 'bulbasaur', 'metapod']



However, this is NOT method overloading, since there is no method to overload (we only have one method to add pokemon to the bag). 

### Method Overloading

We CAN achieve method overloading in Python using the library called multipledispatch:
<!-- (I am gonna ignore the if else handling of cases since it seems inelegant to me) -->


```python
# Source: https://www.geeksforgeeks.org/python-method-overloading/
from multipledispatch import dispatch
class PokeBag:
    def __init__(self):
        self.pokemons = []
    @dispatch(str)
    def add_pokemon(self, pokemon):
        self.pokemons.append(pokemon)
    @dispatch(str, str)
    def add_pokemon(self, pokemon1, pokemon2):
        self.pokemons.append(pokemon1)
        self.pokemons.append(pokemon2)
```


```python
bag = PokeBag()
bag.add_pokemon("ratata")
bag.add_pokemon("arbok", "weezing")
bag.pokemons
```




    ['ratata', 'arbok', 'weezing']



Notice that the code that errored out before doesn't error out when we use the @dispatch decorator from multipledispatch library. So, if we really want to achieve method overloading in Python, we can use the method shown above.

### An exception to the rule

There is one thing that python allows you to overload without using the multipledispatch library and that is defining getters, setters and deleters for you instance properties. These are a bit of an exception however, as they are identified by the decorators used to define them. We'll take a look at these when we get to encapsulation.

I might become a Software Engineer at any second at this rate. Let's move on to the next principle.

## Abstraction

I remember the first time that the word "abstract" was explained to me in English class. It was defined as something intangible, something that you could feel but not touch. Something like love, happiness, sadness or adjectives such as soft or hard. 

When I mention the word- "soft", you already have an idea of "softness" in your head, but not an exact one. I could be referring to a soft pillow, a soft toy, a soft feeling or even a soft person. The word "soft" is abstract and the pillow / toy / feeling / person are concrete examples of abstract concept of the word "soft". All the examples are derived from the same idea of "softness".

Similarly, when it comes to classes we can define an abstract class, which is a class that is not fully formed. It's a blueprint for other classes to be derived from. If we follow the "soft" example, we can define an abstract class called "Soft" and then derive classes like "SoftPillow", "SoftToy", "SoftFeeling" and "SoftPerson" from it.

Let's go back to the notion of abstract classes being the blueprint for classes. If you recall, in the beginning of this post we mentioned that classes are blueprints (for objects). Hence, as  abstract classes are blueprints for classes, they are... blueprints for blueprints. This is why abstract classes are also called meta classes.

### Abtract Classes and abstract methods

In python, to define abstract classes and abstract methods, we need to import the ABC (Abstract Base Class) module from the abc library. Let's look at an example:


```python
from abc import ABC, abstractmethod, abstractproperty
class Soft(ABC):
    @abstractmethod
    def softness(self):
        pass
```

Any python class that derives from the Abstract Base Class (ABC) as shown is an abstract class. It is a class that serves as a blueprint for other classes. It's a blueprint because there isn't an actual real implementation of any functionality in the abstract class. Hence, if you try to create an instance / object of the abstract class, it will error out:


```python
try:
    obj = Soft()
except Exception as e:
    print(f"Exception: {e}")
```

    Exception: Can't instantiate abstract class Soft with abstract method softness
    

This non-instantiation property serves as an easy decision making tool. If you want to allow the creation of parent class objects, use normal inheritance and subclassing. If you do not want to allow the creation of parent class objects, use abstract classes.

Any child class derived from the parent abstract class ("Soft") will need to define the abstract methods mentioned in the parent class. If it doesn't, it will error out. 


```python
class SoftPillow(Soft):
    def not_softness(self):
        print("I am not a soft pillow as I am missing the abstract method softness()!")
```


```python
try:
    obj = SoftPillow()
    print(obj.not_softness())
except Exception as e:
    print(e)
```

    Can't instantiate abstract class SoftPillow with abstract method softness
    

However, if we implement the abstract method softness() in the SoftPillow class, then we can instantiate the class:


```python
class SoftPillow(Soft):
    def softness(self):
        print("I am a super soft Pillow!")
```


```python
obj = SoftPillow()
obj.softness()
```

    I am a super soft Pillow!
    

We could have a mix of abstract and non-abstract methods in an abstract class:


```python
class Soft(ABC):
    @abstractmethod
    def softness(self):
        pass

    def size(self):
        pass
```

The abstract methods are mandatory to be implemented in the child classes, but the non-abstract methods are optional. We illustrate the same below:


```python
class SoftPillow(Soft):
    def size(self):
        print("I am a smol Pillow!")
```


```python
try:
    obj = SoftPillow()
    obj.size()
except Exception as e:
    print(e)
```

    Can't instantiate abstract class SoftPillow with abstract method softness
    

Since we didn't have the mandatory abstract method softness() in the SoftPillow class, it errored out. However, we didn't have to implement the optional non-abstract method size(). 


```python
class Softness(Soft):
    def softness(self):
        print("I am a super soft Pillow!")
```


```python
obj = Softness()
obj.softness()
```

    I am a super soft Pillow!
    

If we implement the mandatory abstract method softness() in the SoftPillow class, then there's no problem.

### Abstract Properties

An abstract class can also have abstract properties and non abtract-properties, just like abstract methods and non-abstract methods. The abstract properties are, surprise, surprise, mandatory to be implemented in the child classes, but the non-abstract properties are optional. Example below:


```python
class Soft(ABC):
    def __init__(self):
        self.price = "Not defined"
    
    @abstractproperty
    def get_price(self):
        pass
```


```python
class SoftToy(Soft):
    def __init__(self):
        self.price = "100"
    
    def get_price(self):
        return self.price
```


```python
obj = SoftToy()
obj.get_price()
```




    '100'



Other than the fact that we're using a different decorator: "@abstractproperty" instead of "@abstractmethod" there isn't a real difference between the two ([Source: StackOverflow](https://stackoverflow.com/questions/14441619/actual-difference-in-implementing-overriding-using-abstractproperty-and-abstra#:~:text=The%20notion%20is%20correct%3B%20the,to%20the%20ABC%20you%20defined.)).

The only convention based nuance is that you would use the "@abstractproperty" decorator when you are dealing with properties and the "@abstractmethod" decorator when you are dealing with methods. 

In the example above, we could have used the "@abstractmethod" decorator instead of the "@abstractproperty" decorator and it would have worked just fine. However, I have used the "@abstractproperty" decorator just so we all know that there is an alternative decorator that exists for properties.

An abstract class can implement some functionality of course, but remember that it cannot be instantiated. It's useful to add implementation to an abstract class if you want to share some common functionality between the child classes. This works in the same way as it does in vanilla inheritance, so we will not look at an example.

## Encapsulation

Encapsulation refers to the process of "encapsulating" data and methods together. When we put the data and methods together in such a capsule, we are able to limit access to the data and methods, so that there's no accidental modification.

There are 3 levels of access modifiers in general, in increasing order of restrictiveness:
- Public
- Protected
- Private

Python has its own conventions when it comes to programming. This is glaringly obvious when we look at access modifiers. If you are a C++ / Java person before and are just discovering Python, it might be a little strange, but you'll get used to it soon enough. 

Typically, in other languages, you would use a keyword to define the access modifier, such as "public", "protected" or "private". For instance in Java:
```
public static void main(String[] args)
```

Python uses underscores to distinguish between public, protected and private methods. Let's take a look at each of these in Python, starting with public properties and methods:

### Public


```python
class Pokemon:
    def __init__(self):
        self.hp = 100
        self.attack = random.randint(0, 100)
        self.defense = random.randint(0, 100)
    
    def speak(self):
        print("I am a Pokemon!")
```

Notice the lack of underscores in the property (attack, defense) and method (speak) names. This is the convention for public properties and methods in Python. Very vanilla, nothing new. Properties are public by default. We can access these properties and methods from outside the class as well as from inside the class.


```python
pokemon = Pokemon()
print(pokemon.attack)
print(pokemon.defense)
pokemon.speak()
```

    92
    81
    I am a Pokemon!
    

In the above examples we accessed the attack, defense and speak properties from outside the class. Hence, they are public. Let's now take a look at protected properties and methods:

### Protected


```python
class Pokemon:
    def __init__(self):
        self._hp = 100
        self._attack = random.randint(0, 100)
        self._defense = random.randint(0, 100)
    
    def _speak(self):
        print("I am a Pokemon!")
```

The instance properties _hp, _attack and _defense as well as the _speak method are meant to be protected. Just like Python doesn't enforce the variable types mentioned as type hints, it also doesn't enforce the access modifiers.

Python buys into the notion that ["We're all adults here"](https://stackoverflow.com/questions/11483366/protected-method-in-python#:~:text=You%20can't.,supposed%20to%20use%20the%20method.&text=By%20convention%2C%20one%20underscore%20is,two%20underscores%20are%20considered%20private.). While the protected methods can be accessed from outside the class, just like the public methods, we as python programmers CHOOSE not to.

This means that protected properties need to be accessed via their getters and setters. And protected methods should only be used internally within a class. Let's look at an example:

You might sometimes encounter a TypeError when using the @property decorator in .ipynb files. I raised an issue about this on the [VS Code Jupyter github repo](https://github.com/microsoft/vscode-jupyter/issues/14748). If you ever encounter the same error, just restart your VS Code and it will be fixed (unless it's a code error). You can also run the code in a .py file and execute the .py file from the command line like an actual software engineer.


```python
class Pokemon():
    def __init__(self):
        self._hp = 100
        self._attack = 0
        self._defense = 0

    @property
    def attack(self): # attack getter
        return self._attack
    
    @attack.setter
    def attack(self, value):
        self._attack = value

    @property
    def defense(self): # defense getter
        return self._defense
    
    @defense.setter
    def defense(self, value):
        self._defense = value

```


```python
pokemon = Pokemon()
print(f"The pokemon's attack stat is: {pokemon.attack}")
print(f"The pokemon's defense stat is: {pokemon.defense}")
pokemon.attack = 50
pokemon.defense = 30
print(f"The pokemon's attack stat is: {pokemon.attack}")
print(f"The pokemon's defense stat is: {pokemon.defense}")
```

    The pokemon's attack stat is: 0
    The pokemon's defense stat is: 0
    The pokemon's attack stat is: 50
    The pokemon's defense stat is: 30
    

Notice that in the above syntax, we do not access the protected properties directly.  Usually, if we had to access a getter function called attack(), we would have called pokemon.attack(). Moreover, if we had a setter function called attack(), we would have set the pokemon's attack using pokemon.attack(some_value). However, since we used the @property decorator,  the function name is treated as a property. When the function is treated as a property, we assign and retrieve values from it as if it were a property. You will be able to contrast this with the non-fancy example that will follow.

In the example above, we define getters and setters for the protected instance properties. We use some slightly fancy syntax to define the getters and setters, but you can choose not to do so and define them in the usual way. If you notice, this is where the Python method overloading exception comes into play. We defined the getters and setters for the protected instance properties using the same method with a different number of arguments. For instance, for the _attack property, we defined the getter with 0 arguments and the setter with 1 argument.

If you do choose to use the fancy way to define getters and setters make sure you follow the order, getter before setter and use the appropriate decorators. Here's a small template for you to use:


```python
@property
def property(self): # property getter
    return self._property

@property.setter
def property(self, value):
    self._property = value
```

The non-fancy version:


```python
class Pokemon():
    def __init__(self):
        self._hp = 100
        self._attack = 0
        self._defense = 0

    def get_attack(self): # attack getter
        return self._attack
    
    def set_attack(self, value):
        self._attack = value    
    
    def get_defense(self): # defense getter
        return self._defense
    
    def set_defense(self, value):
        self._defense = value
```


```python
pokemon = Pokemon()
print(f"The pokemon's attack stat is: {pokemon.get_attack()}")
print(f"The pokemon's defense stat is: {pokemon.get_defense()}")
pokemon.set_attack(50)
pokemon.set_defense(30)
print(f"The pokemon's attack stat is: {pokemon.get_attack()}")
print(f"The pokemon's defense stat is: {pokemon.get_defense()}")
```

    The pokemon's attack stat is: 0
    The pokemon's defense stat is: 0
    The pokemon's attack stat is: 50
    The pokemon's defense stat is: 30
    

While the results are the same, make sure you notice the difference in syntax between the fancy version and the non-fancy version.

### Private

In Python, private properties and methods are defined using double underscores at the beginning of their names. We CANNOT access these private methods and properties from outside the class. Let's look at an example:


```python
class Pokemon:
    def __init__(self):
        self.__hp = 100
        self.__attack = random.randint(0, 100)
        self.defense = random.randint(0, 100)
    
    def __speak(self):
        print("I am a Pokemon!")
```


```python
try:
    pokemon = Pokemon()
    print(pokemon.__hp)
except Exception as e:
    print(e)
```

    'Pokemon' object has no attribute '__hp'
    

The same thing will happen if we try to access the private attack property (__attack) from outside the class. We can compare this with the defense property (note that we use "defense" and not "_defense" for the comparison, as we are not supposed to use the protected property directly):


```python
pokemon = Pokemon()
print(pokemon.defense)
```

    55
    

We cannot access the private methods either:


```python
try:
    pokemon = Pokemon()
    pokemon.__speak()
except Exception as e:
    print(e)
```

    'Pokemon' object has no attribute '__speak'
    

If we really need to access the private methods and properties, for whatever purpose, we can use something called "name mangling". Take a look at the class and the modified way to access the private properties and methods:


```python
class Pokemon:
    def __init__(self):
        self.__hp = 100
        self.__attack = random.randint(0, 100)
        self.defense = random.randint(0, 100)
    
    def __speak(self):
        print("I am a Pokemon!")
```


```python
pokemon = Pokemon()
print(pokemon._Pokemon__hp)
print(pokemon._Pokemon__attack)
```

    100
    17
    


```python
pokemon._Pokemon__speak()
```

    I am a Pokemon!
    

At this point, you should be well and truly prepared for any OOPs questions in your Data Science interviews. We are well and truly on our way to becoming a Software Engineer in addition to being Data Scientists ʕ •ᴥ•ʔ. I do recommend that you practice writing code to practice these principles. If you wanna dig deeper into this rabbit hole (I referred to most of these myself while writing this post):
   - [super()](https://realpython.com/python-super/)
   - [Dive deeper into @property](https://realpython.com/python-property/#creating-read-write-attributes)
   - [Getters and setters](https://realpython.com/python-getter-setter/#using-properties-instead-of-getters-and-setters-the-python-way)
   - [Should you even be writing a class for this?](https://www.youtube.com/watch?v=o9pEzgHorH0)
<!-- #https://twistedsifter.com/2015/06/best-text-emoticons-on-a-single-page/
#https://www.textemoji.org/ -->


