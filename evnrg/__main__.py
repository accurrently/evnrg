import evnrg as evnrg

def main():
    """Prints a description of the package."""

    desc = """ EVNRG: An EV electrical demand and energy simulator.
    Version {}
    """.format(evnrg.__version__)

    print(desc)
    

if __name__ == "__main__":
    main()
