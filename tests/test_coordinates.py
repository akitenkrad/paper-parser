from paper_parser.data import Coordinates, Point


def test_is_intercept():
    c1 = Coordinates(
        top_left=Point(x=0, y=0),
        top_right=Point(x=10, y=0),
        bottom_left=Point(x=0, y=10),
        bottom_right=Point(x=10, y=10),
    )
    c2 = Coordinates(
        top_left=Point(x=5, y=5),
        top_right=Point(x=15, y=5),
        bottom_left=Point(x=5, y=15),
        bottom_right=Point(x=15, y=15),
    )
    assert c1.is_intercept(c2) == True
    assert c2.is_intercept(c1) == True

    c3 = Coordinates(
        top_left=Point(x=11, y=11),
        top_right=Point(x=20, y=11),
        bottom_left=Point(x=11, y=20),
        bottom_right=Point(x=20, y=20),
    )
    assert c1.is_intercept(c3) == False
    assert c3.is_intercept(c1) == False
    assert c2.is_intercept(c3) == True
    assert c3.is_intercept(c2) == True
