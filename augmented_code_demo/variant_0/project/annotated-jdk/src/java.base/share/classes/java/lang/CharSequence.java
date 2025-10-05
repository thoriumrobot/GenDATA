/*
    @Positive
 * Copyright (c) 2000, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.lang;

    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;
    @Positive
import org.checkerframework.checker.index.qual.LengthOf;
    @Positive
import org.checkerframework.checker.index.qual.SameLen;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.checker.nonempty.qual.NonEmpty;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import java.util.Objects;
    @Positive
import java.util.PrimitiveIterator;
    @Positive
import java.util.Spliterator;
    @Positive
import java.util.Spliterators;
    @Positive
import java.util.function.IntConsumer;
    @Positive
import java.util.stream.IntStream;
    @Positive
import java.util.stream.StreamSupport;

    @Positive
@AnnotatedFor({ "lock", "nullness", "index" })
    @Positive
public interface CharSequence {

    @Positive
    @Pure
    @Positive
    @LengthOf({ "this" })
    @Positive
    int length(@GuardSatisfied CharSequence this);

    @Positive
    @Pure
    @Positive
    char charAt(@IndexFor({ "this" }) int index);

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
    default boolean isEmpty();

    @Positive
    @SideEffectFree
    @Positive
    CharSequence subSequence(@IndexOrHigh({ "this" }) int start, @IndexOrHigh({ "this" }) int end);

    @Positive
    @SideEffectFree
    @Positive
    @SameLen({ "this" })
    @Positive
    public String toString(@GuardSatisfied CharSequence this);

    @Positive
    public default IntStream chars();

    @Positive
    public default IntStream codePoints();

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @Pure
    @Positive
    public static int compare(CharSequence cs1, CharSequence cs2);
    @Positive
}

// CFWR semantic augmentation - variant 0
