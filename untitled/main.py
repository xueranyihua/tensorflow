class Solution(object):
    def maxArea(self, height):
        max = 0
        front = 0
        behind = len(height) - 1
        while True:
            if front == behind:
                break
            elif height[front] >= height[behind]:
                area = height[behind]*(behind - front)
                behind -= 1
                if area > max:
                    max = area
            else:
                area = height[front]*(behind - front)
                front += 1
                if area > max:
                    max = area
        return  max