/*
 * Copyright (c) 1997, 2018, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package java.util;

import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
import org.checkerframework.checker.nonempty.qual.NonEmpty;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectsOnly;
import org.checkerframework.framework.qual.AnnotatedFor;

@AnnotatedFor({ "lock", "nullness", "index" })
public interface ListIterator<E> extends Iterator<E> {

    @Pure
    @EnsuresNonEmptyIf(result = true, expression = "this")
    boolean hasNext();

    @SideEffectsOnly("this")
    E next(@GuardSatisfied @NonEmpty ListIterator<E> this);

    @Pure
    boolean hasPrevious();

    E previous(@GuardSatisfied ListIterator<E> this);

    @Pure
    @NonNegative
    int nextIndex();

    @Pure
    @GTENegativeOne
    int previousIndex();

    void remove(@GuardSatisfied ListIterator<E> this);

    void set(@GuardSatisfied ListIterator<E> this, E e);

    void add(@GuardSatisfied ListIterator<E> this, E e);
}
