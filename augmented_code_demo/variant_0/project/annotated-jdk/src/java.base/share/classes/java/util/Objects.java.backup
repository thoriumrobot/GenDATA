/*
 * Copyright (c) 2009, 2020, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.interning.qual.EqualsMethod;
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.nullness.qual.EnsuresNonNull;
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.checker.nullness.qual.PolyNull;
import org.checkerframework.checker.signedness.qual.UnknownSignedness;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import org.checkerframework.framework.qual.CFComment;
import jdk.internal.util.Preconditions;
import jdk.internal.vm.annotation.ForceInline;
import java.util.function.Supplier;

@AnnotatedFor({ "index", "interning", "lock", "nullness" })
@UsesObjectEquals
public final class Objects {

    @Pure
    @EqualsMethod
    public static boolean equals(@GuardSatisfied @Nullable @UnknownSignedness Object a, @GuardSatisfied @Nullable @UnknownSignedness Object b);

    @Pure
    public static boolean deepEquals(@GuardSatisfied @Nullable @UnknownSignedness Object a, @GuardSatisfied @Nullable @UnknownSignedness Object b);

    @Pure
    public static int hashCode(@GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Pure
    public static int hash(@GuardSatisfied @Nullable @UnknownSignedness Object... values);

    @SideEffectFree
    public static String toString(@GuardSatisfied @Nullable @UnknownSignedness Object o);

    @SideEffectFree
    @PolyNull
    public static String toString(@GuardSatisfied @Nullable @UnknownSignedness Object o, @PolyNull String nullDefault);

    @Pure
    public static <T> int compare(@GuardSatisfied @Nullable @UnknownSignedness T a, @GuardSatisfied @Nullable @UnknownSignedness T b, @GuardSatisfied Comparator<? super T> c);

    @CFComment({ "lock: TODO: treat like other nullness assertion methods in the Checker Framework." })
    @EnsuresNonNull("#1")
    @NonNull
    public static <T> T requireNonNull(@NonNull T obj);

    @EnsuresNonNull("#1")
    @SideEffectFree
    @NonNull
    public static <T> T requireNonNull(@GuardSatisfied @NonNull @UnknownSignedness T obj, @Nullable String message);

    @EnsuresNonNullIf(expression = { "#1" }, result = false)
    @Pure
    public static boolean isNull(@GuardSatisfied @Nullable @UnknownSignedness Object obj);

    @EnsuresNonNullIf(expression = { "#1" }, result = true)
    @Pure
    public static boolean nonNull(@GuardSatisfied @Nullable @UnknownSignedness Object obj);

    @NonNull
    public static <T> T requireNonNullElse(@Nullable T obj, @NonNull T defaultObj);

    public static <T extends @NonNull Object> T requireNonNullElseGet(@Nullable T obj, Supplier<? extends T> supplier);

    @EnsuresNonNull("#1")
    @Pure
    @NonNull
    public static <T> T requireNonNull(@GuardSatisfied @NonNull @UnknownSignedness T obj, @GuardSatisfied Supplier<String> messageSupplier);

    @ForceInline
    public static int checkIndex(int index, int length);

    public static int checkFromToIndex(int fromIndex, int toIndex, int length);

    public static int checkFromIndexSize(int fromIndex, int size, int length);

    @ForceInline
    @Pure
    public static long checkIndex(long index, long length);

    @Pure
    public static long checkFromToIndex(long fromIndex, long toIndex, long length);

    @Pure
    public static long checkFromIndexSize(long fromIndex, long size, long length);
}
