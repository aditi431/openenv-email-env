def grade_spam(action, expected):

    if action == expected:
        return 1.0

    return 0.0


def grade_priority(action, expected):

    if action == expected:
        return 1.0

    return 0.0


def grade_routing(category, department, expected):

    score = 0

    if category == expected["category"]:
        score += 0.4

    if department == expected["department"]:
        score += 0.6

    return score