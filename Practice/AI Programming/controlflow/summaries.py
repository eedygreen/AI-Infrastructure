#!/usr/bin/env python
# coding: utf-8

# ## Playground
# Feel free to use the cell below to code.

# In[ ]:





# In[ ]:





# In[ ]:





# ## Conditional Statements
# Use cell below to execute the code, which includes several examples of `if` statements. Experiment with different inputs and see what is printed out. Can you follow the flow of logic to figure out which code lines will get run? If you're not sure, you can insert additional `print` statements to help you figure out how it works.

# In[2]:


#First Example - try changing the value of phone_balance
phone_balance = 10
bank_balance = 50

if phone_balance < 10:
    phone_balance += 10
    bank_balance -= 10

print(phone_balance)
print(bank_balance)

#Second Example - try changing the value of number

number = 145
if number % 2 == 0:
    print("Number " + str(number) + " is even.")
else:
    print("Number " + str(number) + " is odd.")

#Third Example - try to change the value of age
age = 35

# Here are the age limits for bus fares
free_up_to_age = 4
child_up_to_age = 18
senior_from_age = 35

# These lines determine the bus fare prices
concession_ticket = 1.25
adult_ticket = 2.50

# Here is the logic for bus fare prices
if age <= free_up_to_age:
    ticket_price = 0
elif age <= child_up_to_age:
    ticket_price = concession_ticket
elif age >= senior_from_age:
    ticket_price = adult_ticket
else:
    ticket_price = adult_ticket

message = "Somebody who is {} years old will pay ${} to ride the bus.".format(age, ticket_price)
print(message)


# In[ ]:





# ## Iterating Through Dictionaries
# Run and experiment with this example in the code cell below!

# In[3]:


cast = {
           "Jerry Seinfeld": "Jerry Seinfeld",
           "Julia Louis-Dreyfus": "Elaine Benes",
           "Jason Alexander": "George Costanza",
           "Michael Richards": "Cosmo Kramer"
       }

print("Iterating through keys:")
for key in cast:
    print(key)

print("\nIterating through keys and values:")
for key, value in cast.items():
    print("Actor: {}    Role: {}".format(key, value))


# In[ ]:




