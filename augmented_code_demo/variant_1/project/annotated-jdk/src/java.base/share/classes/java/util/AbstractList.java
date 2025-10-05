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
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;
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
@CFComment("lock/nullness: Subclasses of this interface/class may opt to prohibit null elements")
    @Positive
@AnnotatedFor({ "lock", "nullness", "index" })
    @Positive
public abstract class AbstractList<E> extends AbstractCollection<E> implements List<E> {

    @Positive
    @SideEffectFree
    @Positive
    protected AbstractList() {
    @Positive
    }

    @Positive
    @EnsuresNonEmpty("this")
    @Positive
    public boolean add(@GuardSatisfied AbstractList<E> this, E e);

    @Positive
    @Pure
    @Positive
    public abstract E get(@GuardSatisfied AbstractList<E> this, @IndexFor({ "this" }) int index);

    @Positive
    public E set(@GuardSatisfied AbstractList<E> this, @IndexFor({ "this" }) int index, E element);

    @Positive
    public void add(@GuardSatisfied AbstractList<E> this, @IndexOrHigh({ "this" }) int index, E element);

    @Positive
    public E remove(@GuardSatisfied @Shrinkable AbstractList<E> this, @IndexFor({ "this" }) int index);

    @Positive
    @Pure
    @Positive
    @GTENegativeOne
    @Positive
    public int indexOf(@GuardSatisfied AbstractList<E> this, @GuardSatisfied @UnknownSignedness Object o);

    @Positive
    @Pure
    @Positive
    @GTENegativeOne
    @Positive
    public int lastIndexOf(@GuardSatisfied AbstractList<E> this, @GuardSatisfied @UnknownSignedness Object o);

    @Positive
    public void clear(@GuardSatisfied @Shrinkable AbstractList<E> this);

    @Positive
    public boolean addAll(@GuardSatisfied AbstractList<E> this, @IndexOrHigh({ "this" }) int index, Collection<? extends E> c);

    @Positive
    @SideEffectFree
    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public Iterator<E> iterator(@PolyGrowShrink @PolyNonEmpty AbstractList<E> this);

    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public ListIterator<E> listIterator(@PolyGrowShrink @PolyNonEmpty AbstractList<E> this);

    @Positive
    @PolyGrowShrink
    @Positive
    public ListIterator<E> listIterator(@PolyGrowShrink AbstractList<E> this, @IndexOrHigh({ "this" }) final int index);

    @Positive
    private class Itr implements Iterator<E> {

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean hasNext();

    @Positive
        @SideEffectsOnly("this")
    @Positive
        public E next(@NonEmpty Itr this);

    @Positive
        public void remove();

    @Positive
        final void checkForComodification();
    @Positive
    }

    @Positive
    private class ListItr extends Itr implements ListIterator<E> {

    @Positive
        public boolean hasPrevious();

    @Positive
        public E previous();

    @Positive
        public int nextIndex();

    @Positive
        public int previousIndex();

    @Positive
        public void set(E e);

    @Positive
        public void add(E e);
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    @PolyGrowShrink
    @Positive
    public List<E> subList(@GuardSatisfied @PolyGrowShrink AbstractList<E> this, @IndexOrHigh({ "this" }) int fromIndex, @IndexOrHigh({ "this" }) int toIndex);

    @Positive
    static void subListRangeCheck(int fromIndex, int toIndex, int size);

    @Positive
    @Pure
    @Positive
    public boolean equals(@GuardSatisfied AbstractList<E> this, @GuardSatisfied @Nullable Object o);

    @Positive
    @Pure
    @Positive
    public int hashCode(@GuardSatisfied AbstractList<E> this);

    @Positive
    protected void removeRange(@GuardSatisfied @Shrinkable AbstractList<E> this, @IndexOrHigh({ "this" }) int fromIndex, @IndexOrHigh({ "this" }) int toIndex);

    @Positive
    protected transient int modCount;

    @Positive
    static final class RandomAccessSpliterator<E> implements Spliterator<E> {

    @Positive
        public Spliterator<E> trySplit();

    @Positive
        public boolean tryAdvance(Consumer<? super E> action);

    @Positive
        public void forEachRemaining(Consumer<? super E> action);

    @Positive
        public long estimateSize();

    @Positive
        public int characteristics();

    @Positive
        static void checkAbstractListModCount(AbstractList<?> alist, int expectedModCount);
    @Positive
    }

    @Positive
    private static class SubList<E> extends AbstractList<E> {

    @Positive
        protected int size;

    @Positive
        public SubList(AbstractList<E> root, int fromIndex, int toIndex) {
    @Positive
        }

    @Positive
        protected SubList(SubList<E> parent, int fromIndex, int toIndex) {
    @Positive
        }

    @Positive
        public E set(int index, E element);

    @Positive
        public E get(int index);

    @Positive
        @Pure
    @Positive
        public int size();

    @Positive
        public void add(int index, E element);

    @Positive
        public E remove(int index);

    @Positive
        protected void removeRange(int fromIndex, int toIndex);

    @Positive
        public boolean addAll(Collection<? extends E> c);

    @Positive
        public boolean addAll(int index, Collection<? extends E> c);

    @Positive
        public Iterator<E> iterator();

    @Positive
        public ListIterator<E> listIterator(int index);

    @Positive
        public List<E> subList(int fromIndex, int toIndex);
    @Positive
    }

    @Positive
    private static class RandomAccessSubList<E> extends SubList<E> implements RandomAccess {

    @Positive
        public List<E> subList(int fromIndex, int toIndex);
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
