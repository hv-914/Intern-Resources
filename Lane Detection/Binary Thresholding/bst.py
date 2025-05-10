class node:
    def __init__(self, data):
        self.data = data
        self.right = None
        self.left = None

class bst:
    def __init__(self):
        self.root = None

    def insert(self, root, x):
        if root is None:
            return node(x)
        elif x < root.data:
            root.left = self.insert(root.left, x)
        elif x > root.data:
            root.right = self.insert(root.right, x)
        elif x == root.data:
            print('Element is already in the tree')
        return root
    
    def delete(self, root, x):
        if root is None:
            return root
        elif x < root.data:
            root.left = self.delete(root.left, x)
        elif x > root.data:
            root.right = self.delete(root.right, x)
        else:
            if root.left is None:
                return root.right
            elif root.right is None:
                return root.left
            else:
                minNode = root
                while minNode.left:
                    minNode = minNode.left
                
    def preorder(self, root):
        if root:
            print(root.data, end = ' ')
            self.preorder(root.left)
            self.preorder(root.right)
    
    def inorder(self, root):
        if root:
            self.inorder(root.left)
            print(root.data, end = ' ')
            self.inorder(root.right)

    def postorder(self, root):
        if root:
            self.postorder(root.left)
            self.postorder(root.right)
            print(root.data, end = ' ')

def main():
    o = bst()
    while True:
        try:
            n = int(input('Insert: 1, Pre-Order: 2, Post-Order: 3, In-Order: 4, Exit: 99\nEnter Choice: '))
            if n == 1:
                x = int(input('Enter Value: '))
                o.root = o.insert(o.root, x)
            elif n == 2:
                print('Pre-Order Traversal: ')
                o.preorder(o.root)
            elif n == 3:
                print('In-Order Traversal: ')
                o.inorder(o.root)
            elif n == 4:
                print('Post-Order Traversal: ')
                o.postorder(o.root)
            elif n == 99:
                break  # Exit the loop
            else:
                print("Invalid Choice")
        except ValueError:
            print("Invalid input. Please enter a number.")
        print()
        print()
if __name__ == '__main__':
    main()