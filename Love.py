# import random

# def generate_random_numbers():
#     return random.sample(range(1, 100), 6)

# if __name__ == "__main__":
#     random_numbers = generate_random_numbers()
#     print(random_numbers)

import random

numbers = []
for _ in range(6):
    number = random.randint(1, 99)
    numbers.append(number)

print("랜덤한 6개의 숫자:", ", ".join(map(str, numbers)))
