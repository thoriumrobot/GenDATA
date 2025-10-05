/*
    @Positive
 * Copyright (c) 1997, 2020, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.PolyGrowShrink;
    @Positive
import org.checkerframework.checker.index.qual.Shrinkable;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.lock.qual.ReleasesNoLocks;
    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmpty;
    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.checker.nonempty.qual.NonEmpty;
    @Positive
import org.checkerframework.checker.nonempty.qual.PolyNonEmpty;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.nullness.qual.PolyNull;
    @Positive
import org.checkerframework.checker.signedness.qual.PolySigned;
    @Positive
import org.checkerframework.checker.signedness.qual.UnknownSignedness;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.framework.qual.CFComment;
    @Positive
import java.util.function.UnaryOperator;

    @Positive
@CFComment({ "lock/nullness: Subclasses of this interface/class may opt to prohibit null elements" })
    @Positive
@AnnotatedFor({ "lock", "nullness", "index" })
    @Positive
public interface List<E> extends Collection<E> {

    @Positive
    @Pure
    @Positive
    @NonNegative
    @Positive
    int size(@GuardSatisfied List<E> this);

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
    boolean isEmpty(@GuardSatisfied List<E> this);

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
    boolean contains(@GuardSatisfied List<E> this, @UnknownSignedness Object o);

    @Positive
    @SideEffectFree
    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    Iterator<E> iterator(@PolyGrowShrink @PolyNonEmpty List<E> this);

    @Positive
    @SideEffectFree
    @Positive
    @PolyNull
    @Positive
    @PolySigned
    @Positive
    Object[] toArray(List<@PolyNull @PolySigned E> this);

    @Positive
    @SideEffectFree
    @Positive
    @Nullable
    @Positive
    <T extends @UnknownSignedness Object> T[] toArray(@PolyNull T[] a);

    @Positive
    @ReleasesNoLocks
    @Positive
    @SideEffectsOnly("this")
    @Positive
    @EnsuresNonEmpty("this")
    @Positive
    boolean add(@GuardSatisfied List<E> this, E e);

    @Positive
    @SideEffectsOnly("this")
    @Positive
    boolean remove(@GuardSatisfied @Shrinkable List<E> this, @UnknownSignedness Object o);

    @Positive
    @Pure
    @Positive
    boolean containsAll(@GuardSatisfied List<E> this, Collection<? extends @UnknownSignedness Object> c);

    @Positive
    @SideEffectsOnly("this")
    @Positive
    @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
    boolean addAll(@GuardSatisfied List<E> this, Collection<? extends E> c);

    @Positive
    @SideEffectsOnly("this")
    @Positive
    @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
    boolean addAll(@GuardSatisfied List<E> this, @IndexOrHigh({ "this" }) int index, Collection<? extends E> c);

    @Positive
    boolean removeAll(@GuardSatisfied @Shrinkable List<E> this, Collection<? extends @UnknownSignedness Object> c);

    @Positive
    boolean retainAll(@GuardSatisfied @Shrinkable List<E> this, Collection<? extends @UnknownSignedness Object> c);

    @Positive
    default void replaceAll(UnaryOperator<E> operator);

    @Positive
    @SuppressWarnings({ "unchecked", "rawtypes" })
    @Positive
    default void sort(Comparator<? super E> c);

    @Positive
    void clear(@GuardSatisfied @Shrinkable List<E> this);

    @Positive
    @Pure
    @Positive
    boolean equals(@GuardSatisfied List<E> this, @Nullable Object o);

    @Positive
    @Pure
    @Positive
    int hashCode(@GuardSatisfied List<E> this);

    @Positive
    @Pure
    @Positive
    E get(@GuardSatisfied List<E> this, @IndexFor({ "this" }) int index);

    @Positive
    E set(@GuardSatisfied List<E> this, @IndexFor({ "this" }) int index, E element);

    @Positive
    @ReleasesNoLocks
    @Positive
    @SideEffectsOnly("this")
    @Positive
    void add(@GuardSatisfied List<E> this, @IndexOrHigh({ "this" }) int index, E element);

    @Positive
    @ReleasesNoLocks
    @Positive
    E remove(@GuardSatisfied @Shrinkable List<E> this, @IndexFor({ "this" }) int index);

    @Positive
    @Pure
    @Positive
    @GTENegativeOne
    @Positive
    int indexOf(@GuardSatisfied List<E> this, @GuardSatisfied @UnknownSignedness Object o);

    @Positive
    @Pure
    @Positive
    @GTENegativeOne
    @Positive
    int lastIndexOf(@GuardSatisfied List<E> this, @GuardSatisfied @UnknownSignedness Object o);

    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    ListIterator<E> listIterator(@PolyGrowShrink @PolyNonEmpty List<E> this);

    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    ListIterator<E> listIterator(@PolyGrowShrink @PolyNonEmpty List<E> this, @IndexOrHigh({ "this" }) int index);

    @Positive
    @SideEffectFree
    @Positive
    @PolyGrowShrink
    @Positive
    List<E> subList(@GuardSatisfied @PolyGrowShrink List<E> this, @IndexOrHigh({ "this" }) int fromIndex, @IndexOrHigh({ "this" }) int toIndex);

    @Positive
    @SideEffectFree
    @Positive
    @Override
    @Positive
    default Spliterator<E> spliterator();

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    static <E> List<E> of();

    @Positive
    @NonEmpty
    @Positive
    static <E extends Object> List<E> of(E e1);

    @Positive
    @NonEmpty
    @Positive
    static <E extends Object> List<E> of(E e1, E e2);

    @Positive
    @NonEmpty
    @Positive
    static <E extends Object> List<E> of(E e1, E e2, E e3);

    @Positive
    @NonEmpty
    @Positive
    static <E extends Object> List<E> of(E e1, E e2, E e3, E e4);

    @Positive
    @NonEmpty
    @Positive
    static <E extends Object> List<E> of(E e1, E e2, E e3, E e4, E e5);

    @Positive
    @NonEmpty
    @Positive
    static <E extends Object> List<E> of(E e1, E e2, E e3, E e4, E e5, E e6);

    @Positive
    @NonEmpty
    @Positive
    static <E extends Object> List<E> of(E e1, E e2, E e3, E e4, E e5, E e6, E e7);

    @Positive
    @NonEmpty
    @Positive
    static <E extends Object> List<E> of(E e1, E e2, E e3, E e4, E e5, E e6, E e7, E e8);

    @Positive
    @NonEmpty
    @Positive
    static <E extends Object> List<E> of(E e1, E e2, E e3, E e4, E e5, E e6, E e7, E e8, E e9);

    @Positive
    @NonEmpty
    @Positive
    static <E extends Object> List<E> of(E e1, E e2, E e3, E e4, E e5, E e6, E e7, E e8, E e9, E e10);

    @Positive
    @SafeVarargs
    @Positive
    @SuppressWarnings("varargs")
    @Positive
    @PolyNonEmpty
    @Positive
    static <E extends Object> List<E> of(E@PolyNonEmpty ... elements);

    @Positive
    @PolyNonEmpty
    @Positive
    static <E extends Object> List<E> copyOf(@PolyNonEmpty Collection<? extends E> coll);
    @Positive
}

// CFWR semantic augmentation - variant 0
