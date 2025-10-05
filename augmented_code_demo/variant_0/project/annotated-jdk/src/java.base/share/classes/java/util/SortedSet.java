/*
    @Positive
 * Copyright (c) 1998, 2018, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.nonempty.qual.NonEmpty;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
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
public interface SortedSet<E> extends Set<E> {

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    Comparator<? super E> comparator(@GuardSatisfied SortedSet<E> this);

    @Positive
    @SideEffectFree
    @Positive
    SortedSet<E> subSet(@GuardSatisfied SortedSet<E> this, @GuardSatisfied E fromElement, @GuardSatisfied E toElement);

    @Positive
    @SideEffectFree
    @Positive
    SortedSet<E> headSet(@GuardSatisfied SortedSet<E> this, E toElement);

    @Positive
    @SideEffectFree
    @Positive
    SortedSet<E> tailSet(@GuardSatisfied SortedSet<E> this, E fromElement);

    @Positive
    @SideEffectFree
    @Positive
    E first(@GuardSatisfied @NonEmpty SortedSet<E> this);

    @Positive
    @SideEffectFree
    @Positive
    E last(@GuardSatisfied @NonEmpty SortedSet<E> this);

    @Positive
    @Override
    @Positive
    default Spliterator<E> spliterator();
    @Positive
}

// CFWR semantic augmentation - variant 0
