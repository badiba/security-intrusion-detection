import timeit

Scopes = []


def Validate(condition, message):
    if (not condition):
        print(message)


def BeginScope(scopeName):
    Scopes.insert(0, (timeit.default_timer(), scopeName))


def EndScope():
    if (len(Scopes) <= 0):
        print("Trying to end a scope while there is no scope available.")
    else:
        scope = Scopes.pop(0)
        print("Debug: " + scope[1] + " took " +
              str(timeit.default_timer() - scope[0]) + " seconds")
