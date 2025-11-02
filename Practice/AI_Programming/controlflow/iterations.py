Here are some possible solutions for these questions.

Question 1a
Provide a dictionary with the count of nominations for each director.

Here's the logic for my solution:
To solve this question, I use the .items method for dictionaries. Remember, the key in our nominated dictionary is a list of nominated directors. Think Compound Data Structures!
I know I need to create a dictionary where the key is a director and the value is the number of nominations.
But to get each director as a key, I will have to use two for loops.
First, to iterate through the nominated dictionary's value (which here is a list of nominations) .
But I have do to this again to iterate through each element (what I'm trying to get to - a nominated director) in the nominated list.
After that, if the director isn't yet in our dictionary, we give that director a count of one. If the director is in the dictionary, we increment that director's count by one.
Solution code (to be inserted starting in line 9 of the starter code on the previous page):

nom_count_dict = {}
for year, list_dir in nominated.items():
    for director in list_dir:
        if director not in nom_count_dict:
            nom_count_dict[director] = 1
        else:
            nom_count_dict[director] += 1
Question 1b
Provide a dictionary with the count of wins for each director.

Essentially, it is the same logic as above, with the other dictionary.
We could use the same approach as in question 1a and it would work fine, but I've provided a shorter alternative here. Instead of the last 4 lines as above, I've just written 1 line, by using the .get method. In this line, we find the director in the win_count_dict dictionary and return the value for that director (the number of times they've won). If they aren't in the dictionary, get returns 0 for that director. Then we increment that director's count by one.

Solution code (to be inserted starting in line 9 of the starter code on the previous page):

win_count_dict = {}
for year, winnerlist in winners.items():
    for winner in winnerlist:
        win_count_dict[winner] = win_count_dict.get(winner, 0) + 1
Question 2
Provide a list with the name(s) of the director(s) with the most Oscar wins.

Here's the logic for my solution:
To address this question, I will need to first create a dictionary with the number of wins by each winning director. For that I can use the code we wrote for Question 1b above.
#FIRST PART OF SOLUTION
win_count_dict = {}
for year, winnerlist in winners.items():
    for winner in winnerlist:
        win_count_dict[winner] = win_count_dict.get(winner, 0) + 1
This win_count_dict dictionary provides a dictionary with the win count for the directors. We will need this to then identify which key (here, director name) has the highest value (here, win count).
To perform this task, we use a variable highest_count to keep track of the highest count of wins.
We iterate through the dictionary to see if the value for a key (i.e., wins for a director) is more than the highest count.
If it is, we replace that value as the highest_count.
Plus we add that key (here, director name) to our list tracking the most_win_director.
Every time we come upon a value higher than the current highest_count, we replace highest_count with the new higher value, empty the most_win_director and replace it with the new key (i.e., director's name).
#SECOND PART OF SOLUTION
highest_count = 0
most_win_director = []

for key, value in win_count_dict.items():
    if value > highest_count:
        highest_count = value
        most_win_director.clear()
        most_win_director.append(key)
    elif value == highest_count:
        most_win_director.append(key)
    else:
        continue
Here is an alternative compact solution to replace the 12 lines above for the second part of the solution, using the built-in function max(), and a list comprehension with a condition:

#ALTERNATIVE SECOND PART OF SOLUTION
highest_count = max(win_count_dict.values())

most_win_director = [key for key, value in win_count_dict.items() if value == highest_count]