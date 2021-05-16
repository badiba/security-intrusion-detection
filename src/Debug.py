import timeit

Scopes = []
IsDebugActive = False


def EnableDebug():
    global IsDebugActive
    IsDebugActive = True


def DisableDebug():
    global IsDebugActive
    IsDebugActive = False


def PrintDebugMessage(message):
    if (IsDebugActive):
        print(message)


def Validate(condition, message):
    if (not condition):
        PrintDebugMessage(message)


def BeginScope(scopeName):
    Scopes.insert(0, (timeit.default_timer(), scopeName))


def EndScope():
    if (len(Scopes) <= 0):
        PrintDebugMessage(
            "Trying to end a scope while there is no scope available.")
    else:
        scope = Scopes.pop(0)
        PrintDebugMessage("Debug: " + scope[1] + " took " +
                          str(timeit.default_timer() - scope[0]) + " seconds")
