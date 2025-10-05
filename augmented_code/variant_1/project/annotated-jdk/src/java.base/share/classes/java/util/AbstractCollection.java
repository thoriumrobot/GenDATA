/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.PolyGrowShrink;
    @Positive
import org.checkerframework.checker.index.qual.Shrinkable;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
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
import jdk.internal.util.ArraysSupport;

    @Positive
@CFComment("lock/nullness: Subclasses of this interface/class may opt to prohibit null elements")
    @Positive
@AnnotatedFor({ "lock", "nullness", "index" })
    @Positive
public abstract class AbstractCollection<E> implements Collection<E> {

    @Positive
    @SideEffectFree
    @Positive
    protected AbstractCollection() {
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public abstract Iterator<E> iterator(@PolyGrowShrink @PolyNonEmpty AbstractCollection<E> this);

    @Positive
    @Pure
    @Positive
    @NonNegative
    @Positive
    public abstract int size(@GuardSatisfied AbstractCollection<E> this);

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
    public boolean isEmpty(@GuardSatisfied AbstractCollection<E> this);

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
    public boolean contains(@GuardSatisfied AbstractCollection<E> this, @GuardSatisfied @UnknownSignedness Object o);

    @Positive
    @SideEffectFree
    @Positive
    @PolyNull
    @Positive
    @PolySigned
    @Positive
    public Object[] toArray(AbstractCollection<@PolyNull @PolySigned E> this);

    @Positive
    @SideEffectFree
    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @Nullable
    @Positive
    public <T> T[] toArray(@PolyNull T[] a);

    @Positive
    public boolean add(@GuardSatisfied AbstractCollection<E> this, E e);

    @Positive
    public boolean remove(@GuardSatisfied @Shrinkable AbstractCollection<E> this, @GuardSatisfied @UnknownSignedness Object o);

    @Positive
    @Pure
    @Positive
    public boolean containsAll(@GuardSatisfied AbstractCollection<E> this, @GuardSatisfied Collection<? extends @UnknownSignedness Object> c);

    @Positive
    public boolean addAll(@GuardSatisfied AbstractCollection<E> this, Collection<? extends E> c);

    @Positive
    public boolean removeAll(@GuardSatisfied @Shrinkable AbstractCollection<E> this, Collection<? extends @UnknownSignedness Object> c);

    @Positive
    public boolean retainAll(@GuardSatisfied @Shrinkable AbstractCollection<E> this, Collection<? extends @UnknownSignedness Object> c);

    @Positive
    public void clear(@GuardSatisfied @Shrinkable AbstractCollection<E> this);

    @Positive
    @SideEffectFree
    @Positive
    public String toString(@GuardSatisfied AbstractCollection<E> this);
    @Positive
}
