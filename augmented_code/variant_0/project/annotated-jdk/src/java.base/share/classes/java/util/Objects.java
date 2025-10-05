/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2009, 2020, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.interning.qual.EqualsMethod;
    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.nullness.qual.PolyNull;
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
import jdk.internal.util.Preconditions;
    @Positive
import jdk.internal.vm.annotation.ForceInline;
    @Positive
import java.util.function.Supplier;

    @Positive
@AnnotatedFor({ "index", "interning", "lock", "nullness" })
    @Positive
@UsesObjectEquals
    @Positive
public final class Objects {

    @Positive
    @Pure
    @Positive
    @EqualsMethod
    @Positive
    public static boolean equals(@GuardSatisfied @Nullable @UnknownSignedness Object a, @GuardSatisfied @Nullable @UnknownSignedness Object b);

    @Positive
    @Pure
    @Positive
    public static boolean deepEquals(@GuardSatisfied @Nullable @UnknownSignedness Object a, @GuardSatisfied @Nullable @UnknownSignedness Object b);

    @Positive
    @Pure
    @Positive
    public static int hashCode(@GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    @Pure
    @Positive
    public static int hash(@GuardSatisfied @Nullable @UnknownSignedness Object... values);

    @Positive
    @SideEffectFree
    @Positive
    public static String toString(@GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    @SideEffectFree
    @Positive
    @PolyNull
    @Positive
    public static String toString(@GuardSatisfied @Nullable @UnknownSignedness Object o, @PolyNull String nullDefault);

    @Positive
    @Pure
    @Positive
    public static <T> int compare(@GuardSatisfied @Nullable @UnknownSignedness T a, @GuardSatisfied @Nullable @UnknownSignedness T b, @GuardSatisfied Comparator<? super T> c);

    @Positive
    @CFComment({ "lock: TODO: treat like other nullness assertion methods in the Checker Framework." })
    @Positive
    @EnsuresNonNull("#1")
    @Positive
    @NonNull
    @Positive
    public static <T> T requireNonNull(@NonNull T obj);

    @Positive
    @EnsuresNonNull("#1")
    @Positive
    @SideEffectFree
    @Positive
    @NonNull
    @Positive
    public static <T> T requireNonNull(@GuardSatisfied @NonNull @UnknownSignedness T obj, @Nullable String message);

    @Positive
    @EnsuresNonNullIf(expression = { "#1" }, result = false)
    @Positive
    @Pure
    @Positive
    public static boolean isNull(@GuardSatisfied @Nullable @UnknownSignedness Object obj);

    @Positive
    @EnsuresNonNullIf(expression = { "#1" }, result = true)
    @Positive
    @Pure
    @Positive
    public static boolean nonNull(@GuardSatisfied @Nullable @UnknownSignedness Object obj);

    @Positive
    @NonNull
    @Positive
    public static <T> T requireNonNullElse(@Nullable T obj, @NonNull T defaultObj);

    @Positive
    public static <T extends @NonNull Object> T requireNonNullElseGet(@Nullable T obj, Supplier<? extends T> supplier);

    @Positive
    @EnsuresNonNull("#1")
    @Positive
    @Pure
    @Positive
    @NonNull
    @Positive
    public static <T> T requireNonNull(@GuardSatisfied @NonNull @UnknownSignedness T obj, @GuardSatisfied Supplier<String> messageSupplier);

    @Positive
    @ForceInline
    @Positive
    public static int checkIndex(int index, int length);

    @Positive
    public static int checkFromToIndex(int fromIndex, int toIndex, int length);

    @Positive
    public static int checkFromIndexSize(int fromIndex, int size, int length);

    @Positive
    @ForceInline
    @Positive
    @Pure
    @Positive
    public static long checkIndex(long index, long length);

    @Positive
    @Pure
    @Positive
    public static long checkFromToIndex(long fromIndex, long toIndex, long length);

    @Positive
    @Pure
    @Positive
    public static long checkFromIndexSize(long fromIndex, long size, long length);
    @Positive
}
