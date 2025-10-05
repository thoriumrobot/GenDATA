/*
    @Positive
 * Copyright (c) 1997, 2018, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.PolyGrowShrink;
    @Positive
import org.checkerframework.checker.index.qual.Shrinkable;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
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
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.framework.qual.CFComment;
    @Positive
import java.util.function.IntFunction;
    @Positive
import java.util.function.Predicate;
    @Positive
import java.util.stream.Stream;
    @Positive
import java.util.stream.StreamSupport;

    @Positive
@CFComment("lock/nullness: Subclasses of this interface/class may opt to prohibit null elements")
    @Positive
@AnnotatedFor({ "lock", "nullness", "index" })
    @Positive
public interface Collection<E> extends Iterable<E> {

    @Positive
    @Pure
    @Positive
    @NonNegative
    @Positive
    int size(@GuardSatisfied Collection<E> this);

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
    boolean isEmpty(@GuardSatisfied Collection<E> this);

    @Positive
    @CFComment({ "lock: not true, because map could contain nulls:  AssertParametersNonNull(\"get(#1)\")", "Nullness: `o` is not @Nullable because this collection might forbid null", "(though I think a nicer specification would be to return false in that case)" })
    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
    boolean contains(@GuardSatisfied Collection<E> this, @GuardSatisfied @UnknownSignedness Object o);

    @Positive
    @SideEffectFree
    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    Iterator<E> iterator(@PolyGrowShrink @PolyNonEmpty Collection<E> this);

    @Positive
    @CFComment({ "lock: The Nullness Checker does NOT use these signatures for either version", "of toArray; rather, the checker has hard-coded rules for those two", "methods, because the most useful type for toArray is not expressible", "in the surface syntax that the nullness annotations support." })
    @Positive
    @SideEffectFree
    @Positive
    @PolyNull
    @Positive
    @PolySigned
    @Positive
    Object[] toArray(Collection<@PolyNull @PolySigned E> this);

    @Positive
    @SideEffectFree
    @Positive
    @Nullable
    @Positive
    <T extends @UnknownSignedness Object> T[] toArray(@PolyNull T[] a);

    @Positive
    default <T> T[] toArray(IntFunction<T[]> generator);

    @Positive
    @EnsuresNonEmpty("this")
    @Positive
    boolean add(@GuardSatisfied Collection<E> this, E e);

    @Positive
    boolean remove(@GuardSatisfied @Shrinkable Collection<E> this, @UnknownSignedness Object o);

    @Positive
    @Pure
    @Positive
    boolean containsAll(@GuardSatisfied Collection<E> this, @GuardSatisfied Collection<? extends @UnknownSignedness Object> c);

    @Positive
    boolean addAll(@GuardSatisfied Collection<E> this, Collection<? extends E> c);

    @Positive
    boolean removeAll(@GuardSatisfied @Shrinkable Collection<E> this, Collection<? extends @UnknownSignedness Object> c);

    @Positive
    default boolean removeIf(@Shrinkable Collection<E> this, Predicate<? super E> filter);

    @Positive
    boolean retainAll(@GuardSatisfied @Shrinkable Collection<E> this, Collection<? extends @UnknownSignedness Object> c);

    @Positive
    void clear(@GuardSatisfied @Shrinkable Collection<E> this);

    @Positive
    @Pure
    @Positive
    boolean equals(@GuardSatisfied Collection<E> this, @GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    @Pure
    @Positive
    int hashCode(@GuardSatisfied Collection<E> this);

    @Positive
    @SideEffectFree
    @Positive
    @Override
    @Positive
    default Spliterator<E> spliterator();

    @Positive
    @PolyNonEmpty
    @Positive
    default Stream<E> stream(@PolyNonEmpty Collection<E> this);

    @Positive
    default Stream<E> parallelStream();
    @Positive
}

// CFWR semantic augmentation - variant 1
