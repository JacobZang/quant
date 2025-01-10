def calculate_pivot_points(high, low, close):
	pp = (high + low + close) / 3

	r1 = (2 * pp) - low
	s1 = (2 * pp) - high
	r2 = pp + (high - low)
	s2 = pp - (high - low)
	r3 = high + 2 * (pp - low)
	s3 = low - 2 * (high - pp)

	return {
		"Pivot Point": pp,
		"Resistance 1": r1,
		"Support 1": s1,
		"Resistance 2": r2,
		"Support 2": s2,
		"Resistance 3": r3,
		"Support 3": s3
	}

# high = 4503.39
# low = 4391.73
# close = 4414.7

# result = calculate_pivot_points(high, low, close)
# for key, value in result.items():
#     print(f"{key}: {value:.2f}")
