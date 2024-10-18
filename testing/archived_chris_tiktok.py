import heapq


def question(prices: list[int], count: int):
    # Convert prices into a max-heap using negative values
    prices = [-price for price in prices]
    heapq.heapify(prices)

    # Process 'count' iterations
    while count > 0 and prices:
        # Pop the largest value (remember it's negative, so negate it to get the original value)
        largest = -heapq.heappop(prices)

        # Halve the value and insert it back (as a negative value)
        new_value = largest // 2
        heapq.heappush(prices, -new_value)

        count -= 1

    # Return the sum of the remaining prices (convert back to positive)
    return -sum(prices)


# Example usage
prices = [10, 20, 30]
count = 2
print(question(prices, count))  # Output will show the final sum after 'count' iterations
print(question([8, 2, 13], 3))
print(question([9, 1, 5, 3], 4))
print(question([9, 1, 5, 3], 0))
print(question([3, 8, 2, 13], 3))
print(question([], 2))
