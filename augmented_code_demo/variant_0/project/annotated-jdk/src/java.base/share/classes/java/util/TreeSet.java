/*
    @Positive
 * Copyright (c) 1998, 2019, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
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
@CFComment({ "lock/nullness: Subclasses of this interface/class may opt to prohibit null elements" })
    @Positive
@AnnotatedFor({ "lock", "nullness" })
    @Positive
public class TreeSet<E> extends AbstractSet<E> implements NavigableSet<E>, Cloneable, java.io.Serializable {

    @Positive
    public TreeSet() {
    @Positive
    }

    @Positive
    public TreeSet(@Nullable Comparator<? super E> comparator) {
    @Positive
    }

    @Positive
    @PolyNonEmpty
    @Positive
    public TreeSet(@PolyNonEmpty Collection<? extends E> c) {
    @Positive
    }

    @Positive
    @PolyNonEmpty
    @Positive
    public TreeSet(@PolyNonEmpty SortedSet<E> s) {
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public Iterator<E> iterator(@PolyGrowShrink @PolyNonEmpty TreeSet<E> this);

    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public Iterator<E> descendingIterator(@PolyGrowShrink @PolyNonEmpty TreeSet<E> this);

    @Positive
    public NavigableSet<E> descendingSet();

    @Positive
    @Pure
    @Positive
    @NonNegative
    @Positive
    public int size(@GuardSatisfied TreeSet<E> this);

    @Positive
    @EnsuresNonNullIf(expression = { "pollFirst()", "pollLast()" }, result = false)
    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
    public boolean isEmpty(@GuardSatisfied TreeSet<E> this);

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
    public boolean contains(@GuardSatisfied TreeSet<E> this, @GuardSatisfied @UnknownSignedness Object o);

    @Positive
    @EnsuresNonEmpty("this")
    @Positive
    public boolean add(@GuardSatisfied TreeSet<E> this, E e);

    @Positive
    public boolean remove(@GuardSatisfied TreeSet<E> this, @GuardSatisfied @UnknownSignedness Object o);

    @Positive
    public void clear(@GuardSatisfied TreeSet<E> this);

    @Positive
    public boolean addAll(@GuardSatisfied TreeSet<E> this, Collection<? extends E> c);

    @Positive
    @SideEffectFree
    @Positive
    @PolyGrowShrink
    @Positive
    public NavigableSet<E> subSet(@GuardSatisfied @PolyGrowShrink TreeSet<E> this, @GuardSatisfied E fromElement, boolean fromInclusive, @GuardSatisfied E toElement, boolean toInclusive);

    @Positive
    @SideEffectFree
    @Positive
    @PolyGrowShrink
    @Positive
    public NavigableSet<E> headSet(@GuardSatisfied @PolyGrowShrink TreeSet<E> this, @GuardSatisfied E toElement, boolean inclusive);

    @Positive
    @SideEffectFree
    @Positive
    @PolyGrowShrink
    @Positive
    public NavigableSet<E> tailSet(@GuardSatisfied @PolyGrowShrink TreeSet<E> this, @GuardSatisfied E fromElement, boolean inclusive);

    @Positive
    @SideEffectFree
    @Positive
    @PolyGrowShrink
    @Positive
    public SortedSet<E> subSet(@GuardSatisfied @PolyGrowShrink TreeSet<E> this, @GuardSatisfied E fromElement, @GuardSatisfied E toElement);

    @Positive
    @SideEffectFree
    @Positive
    @PolyGrowShrink
    @Positive
    public SortedSet<E> headSet(@GuardSatisfied @PolyGrowShrink TreeSet<E> this, E toElement);

    @Positive
    @SideEffectFree
    @Positive
    @PolyGrowShrink
    @Positive
    public SortedSet<E> tailSet(@GuardSatisfied @PolyGrowShrink TreeSet<E> this, E fromElement);

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public Comparator<? super E> comparator(@GuardSatisfied TreeSet<E> this);

    @Positive
    @SideEffectFree
    @Positive
    public E first(@GuardSatisfied @NonEmpty TreeSet<E> this);

    @Positive
    @SideEffectFree
    @Positive
    public E last(@GuardSatisfied @NonEmpty TreeSet<E> this);

    @Positive
    @Nullable
    @Positive
    public E lower(E e);

    @Positive
    @Nullable
    @Positive
    public E floor(E e);

    @Positive
    @Nullable
    @Positive
    public E ceiling(E e);

    @Positive
    @Nullable
    @Positive
    public E higher(E e);

    @Positive
    @Nullable
    @Positive
    public E pollFirst(@GuardSatisfied TreeSet<E> this);

    @Positive
    @Nullable
    @Positive
    public E pollLast(@GuardSatisfied TreeSet<E> this);

    @Positive
    @SideEffectFree
    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public Object clone(@GuardSatisfied TreeSet<E> this);

    @Positive
    public Spliterator<E> spliterator();
    @Positive
}

// CFWR semantic augmentation - variant 0
