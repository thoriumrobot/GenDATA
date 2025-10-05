/*
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
@AnnotatedFor({ "lock", "nullness", "index" })
    @Positive
public interface Deque<E> extends Queue<E> {

    @Positive
    @EnsuresNonEmpty("this")
    @Positive
    void addFirst(@GuardSatisfied Deque<E> this, E e);

    @Positive
    @EnsuresNonEmpty("this")
    @Positive
    void addLast(@GuardSatisfied Deque<E> this, E e);

    @Positive
    boolean offerFirst(E e);

    @Positive
    boolean offerLast(E e);

    @Positive
    E removeFirst(@GuardSatisfied @NonEmpty @Shrinkable Deque<E> this);

    @Positive
    E removeLast(@GuardSatisfied @NonEmpty @Shrinkable Deque<E> this);

    @Positive
    @Nullable
    @Positive
    E pollFirst(@GuardSatisfied @Shrinkable Deque<E> this);

    @Positive
    @Nullable
    @Positive
    E pollLast(@GuardSatisfied @Shrinkable Deque<E> this);

    @Positive
    @EnsuresNonEmpty("this")
    @Positive
    E getFirst(@GuardSatisfied @NonEmpty @Shrinkable Deque<E> this);

    @Positive
    @EnsuresNonEmpty("this")
    @Positive
    E getLast(@GuardSatisfied @NonEmpty @Shrinkable Deque<E> this);

    @Positive
    @Nullable
    @Positive
    E peekFirst();

    @Positive
    @Nullable
    @Positive
    E peekLast();

    @Positive
    boolean removeFirstOccurrence(@GuardSatisfied @Shrinkable Deque<E> this, Object o);

    @Positive
    boolean removeLastOccurrence(@GuardSatisfied @Shrinkable Deque<E> this, Object o);

    @Positive
    @EnsuresNonEmpty("this")
    @Positive
    boolean add(@GuardSatisfied Deque<E> this, E e);

    @Positive
    boolean offer(E e);

    @Positive
    E remove(@GuardSatisfied @NonEmpty @Shrinkable Deque<E> this);

    @Positive
    @Nullable
    @Positive
    E poll(@GuardSatisfied @Shrinkable Deque<E> this);

    @Positive
    E element(@GuardSatisfied @NonEmpty Deque<E> this);

    @Positive
    @Nullable
    @Positive
    E peek();

    @Positive
    boolean addAll(Collection<? extends E> c);

    @Positive
    void push(@GuardSatisfied Deque<E> this, E e);

    @Positive
    E pop(@GuardSatisfied @NonEmpty @Shrinkable Deque<E> this);

    @Positive
    boolean remove(@GuardSatisfied @Shrinkable Deque<E> this, @UnknownSignedness Object o);

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
    boolean contains(@GuardSatisfied Deque<E> this, @UnknownSignedness Object o);

    @Positive
    @Pure
    @Positive
    @NonNegative
    @Positive
    int size(@GuardSatisfied Deque<E> this);

    @Positive
    @SideEffectFree
    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    Iterator<E> iterator(@PolyGrowShrink @PolyNonEmpty Deque<E> this);

    @Positive
    Iterator<E> descendingIterator();
    @Positive
}

// CFWR semantic augmentation - variant 0
