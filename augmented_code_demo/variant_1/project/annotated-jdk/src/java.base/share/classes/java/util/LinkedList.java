/*
    @Positive
 * Copyright (c) 1997, 2019, Oracle and/or its affiliates. All rights reserved.
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
import java.util.function.Consumer;

    @Positive
@CFComment({ "lock/nullness: This class permits null elements" })
    @Positive
@AnnotatedFor({ "lock", "nullness", "index" })
    @Positive
public class LinkedList<E> extends AbstractSequentialList<E> implements List<E>, Deque<E>, Cloneable, java.io.Serializable {

    @Positive
    public LinkedList() {
    @Positive
    }

    @Positive
    @PolyNonEmpty
    @Positive
    public LinkedList(@PolyNonEmpty Collection<? extends E> c) {
    @Positive
    }

    @Positive
    void linkLast(E e);

    @Positive
    void linkBefore(E e, Node<E> succ);

    @Positive
    E unlink(Node<E> x);

    @Positive
    public E getFirst(@GuardSatisfied @NonEmpty LinkedList<E> this);

    @Positive
    public E getLast(@GuardSatisfied @NonEmpty LinkedList<E> this);

    @Positive
    public E removeFirst(@GuardSatisfied @NonEmpty @Shrinkable LinkedList<E> this);

    @Positive
    public E removeLast(@GuardSatisfied @NonEmpty @Shrinkable LinkedList<E> this);

    @Positive
    public void addFirst(@GuardSatisfied LinkedList<E> this, E e);

    @Positive
    public void addLast(@GuardSatisfied LinkedList<E> this, E e);

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
    public boolean contains(@GuardSatisfied LinkedList<E> this, @GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    @Pure
    @Positive
    @NonNegative
    @Positive
    public int size(@GuardSatisfied LinkedList<E> this);

    @Positive
    @ReleasesNoLocks
    @Positive
    @EnsuresNonEmpty("this")
    @Positive
    public boolean add(@GuardSatisfied LinkedList<E> this, E e);

    @Positive
    @ReleasesNoLocks
    @Positive
    public boolean remove(@GuardSatisfied @Shrinkable LinkedList<E> this, @GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    public boolean addAll(@GuardSatisfied LinkedList<E> this, Collection<? extends E> c);

    @Positive
    public boolean addAll(@GuardSatisfied LinkedList<E> this, @NonNegative int index, Collection<? extends E> c);

    @Positive
    public void clear(@GuardSatisfied @Shrinkable LinkedList<E> this);

    @Positive
    @Pure
    @Positive
    public E get(@GuardSatisfied LinkedList<E> this, @NonNegative int index);

    @Positive
    public E set(@GuardSatisfied LinkedList<E> this, @NonNegative int index, E element);

    @Positive
    public void add(@GuardSatisfied LinkedList<E> this, @NonNegative int index, E element);

    @Positive
    public E remove(@GuardSatisfied @Shrinkable LinkedList<E> this, @NonNegative int index);

    @Positive
    Node<E> node(@NonNegative int index);

    @Positive
    @Pure
    @Positive
    @GTENegativeOne
    @Positive
    public int indexOf(@GuardSatisfied LinkedList<E> this, @GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    @Pure
    @Positive
    @GTENegativeOne
    @Positive
    public int lastIndexOf(@GuardSatisfied LinkedList<E> this, @GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public E peek();

    @Positive
    public E element(@GuardSatisfied @NonEmpty LinkedList<E> this);

    @Positive
    @Nullable
    @Positive
    public E poll(@GuardSatisfied @Shrinkable LinkedList<E> this);

    @Positive
    public E remove(@GuardSatisfied @NonEmpty @Shrinkable LinkedList<E> this);

    @Positive
    public boolean offer(E e);

    @Positive
    public boolean offerFirst(E e);

    @Positive
    public boolean offerLast(E e);

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public E peekFirst();

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public E peekLast();

    @Positive
    @Nullable
    @Positive
    public E pollFirst(@GuardSatisfied @Shrinkable LinkedList<E> this);

    @Positive
    @Nullable
    @Positive
    public E pollLast(@GuardSatisfied @Shrinkable LinkedList<E> this);

    @Positive
    public void push(@GuardSatisfied LinkedList<E> this, E e);

    @Positive
    public E pop(@GuardSatisfied @NonEmpty @Shrinkable LinkedList<E> this);

    @Positive
    public boolean removeFirstOccurrence(@GuardSatisfied @Shrinkable LinkedList<E> this, @GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    public boolean removeLastOccurrence(@GuardSatisfied @Shrinkable LinkedList<E> this, @GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public ListIterator<E> listIterator(@PolyGrowShrink @PolyNonEmpty LinkedList<E> this, @NonNegative int index);

    @Positive
    private class ListItr implements ListIterator<E> {

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean hasNext();

    @Positive
        @SideEffectsOnly("this")
    @Positive
        public E next(@NonEmpty ListItr this);

    @Positive
        public boolean hasPrevious();

    @Positive
        public E previous();

    @Positive
        public int nextIndex();

    @Positive
        public int previousIndex();

    @Positive
        public void remove();

    @Positive
        public void set(E e);

    @Positive
        public void add(E e);

    @Positive
        public void forEachRemaining(Consumer<? super E> action);

    @Positive
        final void checkForComodification();
    @Positive
    }

    @Positive
    private static class Node<E> {
    @Positive
    }

    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public Iterator<E> descendingIterator(@PolyGrowShrink @PolyNonEmpty LinkedList<E> this);

    @Positive
    private class DescendingIterator implements Iterator<E> {

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean hasNext();

    @Positive
        @SideEffectsOnly("this")
    @Positive
        public E next(@NonEmpty DescendingIterator this);

    @Positive
        public void remove();
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    public Object clone(@GuardSatisfied LinkedList<E> this);

    @Positive
    @SideEffectFree
    @Positive
    @PolyNull
    @Positive
    @PolySigned
    @Positive
    public Object[] toArray(LinkedList<@PolyNull @PolySigned E> this);

    @Positive
    @SideEffectFree
    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @Nullable
    @Positive
    public <T> T[] toArray(@PolyNull T[] a);

    @Positive
    @SideEffectFree
    @Positive
    @Override
    @Positive
    public Spliterator<E> spliterator();

    @Positive
    static final class LLSpliterator<E> implements Spliterator<E> {

    @Positive
        final int getEst();

    @Positive
        public long estimateSize();

    @Positive
        public Spliterator<E> trySplit();

    @Positive
        public void forEachRemaining(Consumer<? super E> action);

    @Positive
        public boolean tryAdvance(Consumer<? super E> action);

    @Positive
        public int characteristics();
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
