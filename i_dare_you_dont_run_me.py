# hehe rate the one liner | do not mark this code, its for real for fun
# Little piece of code will create a christmas tree for you ! With an epic animé quote ofcourse
(
    lambda width: print(
        '\033[0;32;40m{tree}\n{root}\n\n\033[5;31;40m{message}\n\033[0m'.format(
            tree='\n'.join('{}'.format(('★' * stars).center(width, '.')) for stars in range(1, width, 2)),
            root='▓'.center(width),
            message=(lambda quote: "".join([quote['character'], ": ", quote['quote']]))(
                __import__('random').choice(
                    __import__('json').loads(
                        __import__('requests').get('https://animechan.vercel.app/api/quotes').text))).center(width))))(
                            2 * int(input('Tree Size: ')))
# Its made to be as stupidly complex and dumb as possible

print("Now that you ran me, you kinda have to look at the code base for this file ? [do not do it]")
