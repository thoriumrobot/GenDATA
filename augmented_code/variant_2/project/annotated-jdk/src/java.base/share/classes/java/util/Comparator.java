/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package java.util;

    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.framework.qual.CFComment;
    @Positive
import java.io.Serializable;
    @Positive
import java.util.function.Function;
    @Positive
import java.util.function.ToIntFunction;
    @Positive
import java.util.function.ToLongFunction;
    @Positive
import java.util.function.ToDoubleFunction;
    @Positive
import java.util.Comparators;

    @Positive
@CFComment({ "lock/nullness: Javadoc says: \"a comparator may optionally permit comparison of null", "arguments, while maintaining the requirements for an equivalence relation.\"" })
    @Positive
@AnnotatedFor({ "lock", "nullness", "index" })
    @Positive
@FunctionalInterface
    @Positive
public interface Comparator<T> {

    @Positive
    int compare(T o1, T o2);

    @Positive
    @Pure
    @Positive
    boolean equals(@GuardSatisfied Comparator<T> this, @GuardSatisfied @Nullable Object obj);

    @Positive
    default Comparator<T> reversed();

    @Positive
    default Comparator<T> thenComparing(Comparator<? super T> other);

    @Positive
    default <U> Comparator<T> thenComparing(Function<? super T, ? extends U> keyExtractor, Comparator<? super U> keyComparator);

    @Positive
    default <U extends Comparable<? super U>> Comparator<T> thenComparing(Function<? super T, ? extends U> keyExtractor);

    @Positive
    default Comparator<T> thenComparingInt(ToIntFunction<? super T> keyExtractor);

    @Positive
    default Comparator<T> thenComparingLong(ToLongFunction<? super T> keyExtractor);

    @Positive
    default Comparator<T> thenComparingDouble(ToDoubleFunction<? super T> keyExtractor);

    @Positive
    public static <T extends Comparable<? super T>> Comparator<T> reverseOrder();

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public static <T extends Comparable<@NonNull ? super @NonNull T>> Comparator<T> naturalOrder();

    @Positive
    public static <T> Comparator<@Nullable T> nullsFirst(Comparator<@Nullable ? super T> comparator);

    @Positive
    public static <T> Comparator<@Nullable T> nullsLast(Comparator<@Nullable ? super T> comparator);

    @Positive
    public static <T, U> Comparator<T> comparing(Function<? super T, ? extends U> keyExtractor, Comparator<? super U> keyComparator);

    @Positive
    public static <T, U extends Comparable<? super U>> Comparator<T> comparing(Function<? super T, ? extends U> keyExtractor);

    @Positive
    public static <T> Comparator<T> comparingInt(ToIntFunction<? super T> keyExtractor);

    @Positive
    public static <T> Comparator<T> comparingLong(ToLongFunction<? super T> keyExtractor);

    @Positive
    public static <T> Comparator<T> comparingDouble(ToDoubleFunction<? super T> keyExtractor);
    @Positive
}
