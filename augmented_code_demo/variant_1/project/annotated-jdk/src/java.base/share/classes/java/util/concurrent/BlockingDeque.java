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
package java.util.concurrent;

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
import org.checkerframework.checker.nullness.qual.NonNull;
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
import java.util.Deque;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.NoSuchElementException;

    @Positive
@AnnotatedFor({ "nullness" })
    @Positive
public interface BlockingDeque<E extends @NonNull Object> extends BlockingQueue<E>, Deque<E> {

    @Positive
    void addFirst(E e);

    @Positive
    void addLast(E e);

    @Positive
    boolean offerFirst(E e);

    @Positive
    boolean offerLast(E e);

    @Positive
    void putFirst(E e) throws InterruptedException;

    @Positive
    void putLast(E e) throws InterruptedException;

    @Positive
    boolean offerFirst(E e, long timeout, TimeUnit unit) throws InterruptedException;

    @Positive
    boolean offerLast(E e, long timeout, TimeUnit unit) throws InterruptedException;

    @Positive
    E takeFirst(@Shrinkable BlockingDeque<E> this) throws InterruptedException;

    @Positive
    E takeLast(@Shrinkable BlockingDeque<E> this) throws InterruptedException;

    @Positive
    @Nullable
    @Positive
    E pollFirst(@Shrinkable BlockingDeque<E> this, long timeout, TimeUnit unit) throws InterruptedException;

    @Positive
    @Nullable
    @Positive
    E pollLast(@Shrinkable BlockingDeque<E> this, long timeout, TimeUnit unit) throws InterruptedException;

    @Positive
    boolean removeFirstOccurrence(@Shrinkable BlockingDeque<E> this, Object o);

    @Positive
    boolean removeLastOccurrence(@Shrinkable BlockingDeque<E> this, Object o);

    @Positive
    @EnsuresNonEmpty("this")
    @Positive
    boolean add(E e);

    @Positive
    boolean offer(E e);

    @Positive
    void put(E e) throws InterruptedException;

    @Positive
    boolean offer(E e, long timeout, TimeUnit unit) throws InterruptedException;

    @Positive
    E remove(@GuardSatisfied @NonEmpty @Shrinkable BlockingDeque<E> this);

    @Positive
    @Nullable
    @Positive
    E poll(@Shrinkable BlockingDeque<E> this);

    @Positive
    E take(@Shrinkable BlockingDeque<E> this) throws InterruptedException;

    @Positive
    @Nullable
    @Positive
    E poll(@Shrinkable BlockingDeque<E> this, long timeout, TimeUnit unit) throws InterruptedException;

    @Positive
    E element(@NonEmpty BlockingDeque<E> this);

    @Positive
    @Nullable
    @Positive
    E peek();

    @Positive
    boolean remove(@Shrinkable BlockingDeque<E> this, @UnknownSignedness Object o);

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
    boolean contains(@UnknownSignedness Object o);

    @Positive
    @Pure
    @Positive
    int size();

    @Positive
    @SideEffectFree
    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    Iterator<E> iterator(@PolyGrowShrink @PolyNonEmpty BlockingDeque<E> this);

    @Positive
    void push(E e);
    @Positive
}

// CFWR semantic augmentation - variant 1
