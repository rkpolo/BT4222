import random

import numpy as np
import pytest

import pandas as pd
from pandas import Categorical, DataFrame, NaT, Timestamp, date_range
import pandas._testing as tm


class TestDataFrameSortValues:
    def test_sort_values(self):
        frame = DataFrame(
            [[1, 1, 2], [3, 1, 0], [4, 5, 6]], index=[1, 2, 3], columns=list("ABC")
        )

        # by column (axis=0)
        sorted_df = frame.sort_values(by="A")
        indexer = frame["A"].argsort().values
        expected = frame.loc[frame.index[indexer]]
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.sort_values(by="A", ascending=False)
        indexer = indexer[::-1]
        expected = frame.loc[frame.index[indexer]]
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.sort_values(by="A", ascending=False)
        tm.assert_frame_equal(sorted_df, expected)

        # GH4839
        sorted_df = frame.sort_values(by=["A"], ascending=[False])
        tm.assert_frame_equal(sorted_df, expected)

        # multiple bys
        sorted_df = frame.sort_values(by=["B", "C"])
        expected = frame.loc[[2, 1, 3]]
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.sort_values(by=["B", "C"], ascending=False)
        tm.assert_frame_equal(sorted_df, expected[::-1])

        sorted_df = frame.sort_values(by=["B", "A"], ascending=[True, False])
        tm.assert_frame_equal(sorted_df, expected)

        msg = "No axis named 2 for object type <class 'pandas.core.frame.DataFrame'>"
        with pytest.raises(ValueError, match=msg):
            frame.sort_values(by=["A", "B"], axis=2, inplace=True)

        # by row (axis=1): GH#10806
        sorted_df = frame.sort_values(by=3, axis=1)
        expected = frame
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.sort_values(by=3, axis=1, ascending=False)
        expected = frame.reindex(columns=["C", "B", "A"])
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.sort_values(by=[1, 2], axis="columns")
        expected = frame.reindex(columns=["B", "A", "C"])
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.sort_values(by=[1, 3], axis=1, ascending=[True, False])
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.sort_values(by=[1, 3], axis=1, ascending=False)
        expected = frame.reindex(columns=["C", "B", "A"])
        tm.assert_frame_equal(sorted_df, expected)

        msg = r"Length of ascending \(5\) != length of by \(2\)"
        with pytest.raises(ValueError, match=msg):
            frame.sort_values(by=["A", "B"], axis=0, ascending=[True] * 5)

    def test_sort_values_inplace(self):
        frame = DataFrame(
            np.random.randn(4, 4), index=[1, 2, 3, 4], columns=["A", "B", "C", "D"]
        )

        sorted_df = frame.copy()
        sorted_df.sort_values(by="A", inplace=True)
        expected = frame.sort_values(by="A")
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.copy()
        sorted_df.sort_values(by=1, axis=1, inplace=True)
        expected = frame.sort_values(by=1, axis=1)
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.copy()
        sorted_df.sort_values(by="A", ascending=False, inplace=True)
        expected = frame.sort_values(by="A", ascending=False)
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.copy()
        sorted_df.sort_values(by=["A", "B"], ascending=False, inplace=True)
        expected = frame.sort_values(by=["A", "B"], ascending=False)
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_values_multicolumn(self):
        A = np.arange(5).repeat(20)
        B = np.tile(np.arange(5), 20)
        random.shuffle(A)
        random.shuffle(B)
        frame = DataFrame({"A": A, "B": B, "C": np.random.randn(100)})

        result = frame.sort_values(by=["A", "B"])
        indexer = np.lexsort((frame["B"], frame["A"]))
        expected = frame.take(indexer)
        tm.assert_frame_equal(result, expected)

        result = frame.sort_values(by=["A", "B"], ascending=False)
        indexer = np.lexsort(
            (frame["B"].rank(ascending=False), frame["A"].rank(ascending=False))
        )
        expected = frame.take(indexer)
        tm.assert_frame_equal(result, expected)

        result = frame.sort_values(by=["B", "A"])
        indexer = np.lexsort((frame["A"], frame["B"]))
        expected = frame.take(indexer)
        tm.assert_frame_equal(result, expected)

    def test_sort_values_multicolumn_uint64(self):
        # GH#9918
        # uint64 multicolumn sort

        df = pd.DataFrame(
            {
                "a": pd.Series([18446637057563306014, 1162265347240853609]),
                "b": pd.Series([1, 2]),
            }
        )
        df["a"] = df["a"].astype(np.uint64)
        result = df.sort_values(["a", "b"])

        expected = pd.DataFrame(
            {
                "a": pd.Series([18446637057563306014, 1162265347240853609]),
                "b": pd.Series([1, 2]),
            },
            index=pd.Index([1, 0]),
        )

        tm.assert_frame_equal(result, expected)

    def test_sort_values_nan(self):
        # GH#3917
        df = DataFrame(
            {"A": [1, 2, np.nan, 1, 6, 8, 4], "B": [9, np.nan, 5, 2, 5, 4, 5]}
        )

        # sort one column only
        expected = DataFrame(
            {"A": [np.nan, 1, 1, 2, 4, 6, 8], "B": [5, 9, 2, np.nan, 5, 5, 4]},
            index=[2, 0, 3, 1, 6, 4, 5],
        )
        sorted_df = df.sort_values(["A"], na_position="first")
        tm.assert_frame_equal(sorted_df, expected)

        expected = DataFrame(
            {"A": [np.nan, 8, 6, 4, 2, 1, 1], "B": [5, 4, 5, 5, np.nan, 9, 2]},
            index=[2, 5, 4, 6, 1, 0, 3],
        )
        sorted_df = df.sort_values(["A"], na_position="first", ascending=False)
        tm.assert_frame_equal(sorted_df, expected)

        expected = df.reindex(columns=["B", "A"])
        sorted_df = df.sort_values(by=1, axis=1, na_position="first")
        tm.assert_frame_equal(sorted_df, expected)

        # na_position='last', order
        expected = DataFrame(
            {"A": [1, 1, 2, 4, 6, 8, np.nan], "B": [2, 9, np.nan, 5, 5, 4, 5]},
            index=[3, 0, 1, 6, 4, 5, 2],
        )
        sorted_df = df.sort_values(["A", "B"])
        tm.assert_frame_equal(sorted_df, expected)

        # na_position='first', order
        expected = DataFrame(
            {"A": [np.nan, 1, 1, 2, 4, 6, 8], "B": [5, 2, 9, np.nan, 5, 5, 4]},
            index=[2, 3, 0, 1, 6, 4, 5],
        )
        sorted_df = df.sort_values(["A", "B"], na_position="first")
        tm.assert_frame_equal(sorted_df, expected)

        # na_position='first', not order
        expected = DataFrame(
            {"A": [np.nan, 1, 1, 2, 4, 6, 8], "B": [5, 9, 2, np.nan, 5, 5, 4]},
            index=[2, 0, 3, 1, 6, 4, 5],
        )
        sorted_df = df.sort_values(["A", "B"], ascending=[1, 0], na_position="first")
        tm.assert_frame_equal(sorted_df, expected)

        # na_position='last', not order
        expected = DataFrame(
            {"A": [8, 6, 4, 2, 1, 1, np.nan], "B": [4, 5, 5, np.nan, 2, 9, 5]},
            index=[5, 4, 6, 1, 3, 0, 2],
        )
        sorted_df = df.sort_values(["A", "B"], ascending=[0, 1], na_position="last")
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_values_stable_descending_sort(self):
        # GH#6399
        df = DataFrame(
            [[2, "first"], [2, "second"], [1, "a"], [1, "b"]],
            columns=["sort_col", "order"],
        )
        sorted_df = df.sort_values(by="sort_col", kind="mergesort", ascending=False)
        tm.assert_frame_equal(df, sorted_df)

    def test_sort_values_stable_descending_multicolumn_sort(self):
        df = DataFrame(
            {"A": [1, 2, np.nan, 1, 6, 8, 4], "B": [9, np.nan, 5, 2, 5, 4, 5]}
        )
        # test stable mergesort
        expected = DataFrame(
            {"A": [np.nan, 8, 6, 4, 2, 1, 1], "B": [5, 4, 5, 5, np.nan, 2, 9]},
            index=[2, 5, 4, 6, 1, 3, 0],
        )
        sorted_df = df.sort_values(
            ["A", "B"], ascending=[0, 1], na_position="first", kind="mergesort"
        )
        tm.assert_frame_equal(sorted_df, expected)

        expected = DataFrame(
            {"A": [np.nan, 8, 6, 4, 2, 1, 1], "B": [5, 4, 5, 5, np.nan, 9, 2]},
            index=[2, 5, 4, 6, 1, 0, 3],
        )
        sorted_df = df.sort_values(
            ["A", "B"], ascending=[0, 0], na_position="first", kind="mergesort"
        )
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_values_stable_categorial(self):
        # GH#16793
        df = DataFrame({"x": pd.Categorical(np.repeat([1, 2, 3, 4], 5), ordered=True)})
        expected = df.copy()
        sorted_df = df.sort_values("x", kind="mergesort")
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_values_datetimes(self):

        # GH#3461, argsort / lexsort differences for a datetime column
        df = DataFrame(
            ["a", "a", "a", "b", "c", "d", "e", "f", "g"],
            columns=["A"],
            index=date_range("20130101", periods=9),
        )
        dts = [
            Timestamp(x)
            for x in [
                "2004-02-11",
                "2004-01-21",
                "2004-01-26",
                "2005-09-20",
                "2010-10-04",
                "2009-05-12",
                "2008-11-12",
                "2010-09-28",
                "2010-09-28",
            ]
        ]
        df["B"] = dts[::2] + dts[1::2]
        df["C"] = 2.0
        df["A1"] = 3.0

        df1 = df.sort_values(by="A")
        df2 = df.sort_values(by=["A"])
        tm.assert_frame_equal(df1, df2)

        df1 = df.sort_values(by="B")
        df2 = df.sort_values(by=["B"])
        tm.assert_frame_equal(df1, df2)

        df1 = df.sort_values(by="B")

        df2 = df.sort_values(by=["C", "B"])
        tm.assert_frame_equal(df1, df2)

    def test_sort_values_frame_column_inplace_sort_exception(self, float_frame):
        s = float_frame["A"]
        with pytest.raises(ValueError, match="This Series is a view"):
            s.sort_values(inplace=True)

        cp = s.copy()
        cp.sort_values()  # it works!

    def test_sort_values_nat_values_in_int_column(self):

        # GH#14922: "sorting with large float and multiple columns incorrect"

        # cause was that the int64 value NaT was considered as "na". Which is
        # only correct for datetime64 columns.

        int_values = (2, int(NaT))
        float_values = (2.0, -1.797693e308)

        df = DataFrame(
            dict(int=int_values, float=float_values), columns=["int", "float"]
        )

        df_reversed = DataFrame(
            dict(int=int_values[::-1], float=float_values[::-1]),
            columns=["int", "float"],
            index=[1, 0],
        )

        # NaT is not a "na" for int64 columns, so na_position must not
        # influence the result:
        df_sorted = df.sort_values(["int", "float"], na_position="last")
        tm.assert_frame_equal(df_sorted, df_reversed)

        df_sorted = df.sort_values(["int", "float"], na_position="first")
        tm.assert_frame_equal(df_sorted, df_reversed)

        # reverse sorting order
        df_sorted = df.sort_values(["int", "float"], ascending=False)
        tm.assert_frame_equal(df_sorted, df)

        # and now check if NaT is still considered as "na" for datetime64
        # columns:
        df = DataFrame(
            dict(datetime=[Timestamp("2016-01-01"), NaT], float=float_values),
            columns=["datetime", "float"],
        )

        df_reversed = DataFrame(
            dict(datetime=[NaT, Timestamp("2016-01-01")], float=float_values[::-1]),
            columns=["datetime", "float"],
            index=[1, 0],
        )

        df_sorted = df.sort_values(["datetime", "float"], na_position="first")
        tm.assert_frame_equal(df_sorted, df_reversed)

        df_sorted = df.sort_values(["datetime", "float"], na_position="last")
        tm.assert_frame_equal(df_sorted, df)

        # Ascending should not affect the results.
        df_sorted = df.sort_values(["datetime", "float"], ascending=False)
        tm.assert_frame_equal(df_sorted, df)

    def test_sort_values_na_position_with_categories(self):
        # GH#22556
        # Positioning missing value properly when column is Categorical.
        categories = ["A", "B", "C"]
        category_indices = [0, 2, 4]
        list_of_nans = [np.nan, np.nan]
        na_indices = [1, 3]
        na_position_first = "first"
        na_position_last = "last"
        column_name = "c"

        reversed_categories = sorted(categories, reverse=True)
        reversed_category_indices = sorted(category_indices, reverse=True)
        reversed_na_indices = sorted(na_indices)

        df = pd.DataFrame(
            {
                column_name: pd.Categorical(
                    ["A", np.nan, "B", np.nan, "C"], categories=categories, ordered=True
                )
            }
        )
        # sort ascending with na first
        result = df.sort_values(
            by=column_name, ascending=True, na_position=na_position_first
        )
        expected = DataFrame(
            {
                column_name: Categorical(
                    list_of_nans + categories, categories=categories, ordered=True
                )
            },
            index=na_indices + category_indices,
        )

        tm.assert_frame_equal(result, expected)

        # sort ascending with na last
        result = df.sort_values(
            by=column_name, ascending=True, na_position=na_position_last
        )
        expected = DataFrame(
            {
                column_name: Categorical(
                    categories + list_of_nans, categories=categories, ordered=True
                )
            },
            index=category_indices + na_indices,
        )

        tm.assert_frame_equal(result, expected)

        # sort descending with na first
        result = df.sort_values(
            by=column_name, ascending=False, na_position=na_position_first
        )
        expected = DataFrame(
            {
                column_name: Categorical(
                    list_of_nans + reversed_categories,
                    categories=categories,
                    ordered=True,
                )
            },
            index=reversed_na_indices + reversed_category_indices,
        )

        tm.assert_frame_equal(result, expected)

        # sort descending with na last
        result = df.sort_values(
            by=column_name, ascending=False, na_position=na_position_last
        )
        expected = DataFrame(
            {
                column_name: Categorical(
                    reversed_categories + list_of_nans,
                    categories=categories,
                    ordered=True,
                )
            },
            index=reversed_category_indices + reversed_na_indices,
        )

        tm.assert_frame_equal(result, expected)

    def test_sort_values_nat(self):

        # GH#16836

        d1 = [Timestamp(x) for x in ["2016-01-01", "2015-01-01", np.nan, "2016-01-01"]]
        d2 = [
            Timestamp(x)
            for x in ["2017-01-01", "2014-01-01", "2016-01-01", "2015-01-01"]
        ]
        df = pd.DataFrame({"a": d1, "b": d2}, index=[0, 1, 2, 3])

        d3 = [Timestamp(x) for x in ["2015-01-01", "2016-01-01", "2016-01-01", np.nan]]
        d4 = [
            Timestamp(x)
            for x in ["2014-01-01", "2015-01-01", "2017-01-01", "2016-01-01"]
        ]
        expected = pd.DataFrame({"a": d3, "b": d4}, index=[1, 3, 0, 2])
        sorted_df = df.sort_values(by=["a", "b"])
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_values_na_position_with_categories_raises(self):
        df = pd.DataFrame(
            {
                "c": pd.Categorical(
                    ["A", np.nan, "B", np.nan, "C"],
                    categories=["A", "B", "C"],
                    ordered=True,
                )
            }
        )

        with pytest.raises(ValueError):
            df.sort_values(by="c", ascending=False, na_position="bad_position")

    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize(
        "original_dict, sorted_dict, ignore_index, output_index",
        [
            ({"A": [1, 2, 3]}, {"A": [3, 2, 1]}, True, [0, 1, 2]),
            ({"A": [1, 2, 3]}, {"A": [3, 2, 1]}, False, [2, 1, 0]),
            (
                {"A": [1, 2, 3], "B": [2, 3, 4]},
                {"A": [3, 2, 1], "B": [4, 3, 2]},
                True,
                [0, 1, 2],
            ),
            (
                {"A": [1, 2, 3], "B": [2, 3, 4]},
                {"A": [3, 2, 1], "B": [4, 3, 2]},
                False,
                [2, 1, 0],
            ),
        ],
    )
    def test_sort_values_ignore_index(
        self, inplace, original_dict, sorted_dict, ignore_index, output_index
    ):
        # GH 30114
        df = DataFrame(original_dict)
        expected = DataFrame(sorted_dict, index=output_index)
        kwargs = {"ignore_index": ignore_index, "inplace": inplace}

        if inplace:
            result_df = df.copy()
            result_df.sort_values("A", ascending=False, **kwargs)
        else:
            result_df = df.sort_values("A", ascending=False, **kwargs)

        tm.assert_frame_equal(result_df, expected)
        tm.assert_frame_equal(df, DataFrame(original_dict))

    def test_sort_values_nat_na_position_default(self):
        # GH 13230
        expected = pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 4],
                "date": pd.DatetimeIndex(
                    [
                        "2010-01-01 09:00:00",
                        "2010-01-01 09:00:01",
                        "2010-01-01 09:00:02",
                        "2010-01-01 09:00:03",
                        "NaT",
                    ]
                ),
            }
        )
        result = expected.sort_values(["A", "date"])
        tm.assert_frame_equal(result, expected)
