/*
    @Positive
 * Copyright (c) 1994, 2020, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.checker.lock.qual.NewObject;
    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.common.value.qual.PolyValue;
    @Positive
import org.checkerframework.common.value.qual.StaticallyExecutable;
    @Positive
import org.checkerframework.common.value.qual.StringVal;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import jdk.internal.vm.annotation.IntrinsicCandidate;
    @Positive
import java.lang.constant.Constable;
    @Positive
import java.lang.constant.ConstantDesc;
    @Positive
import java.lang.constant.ConstantDescs;
    @Positive
import java.lang.constant.DynamicConstantDesc;
    @Positive
import java.util.Optional;
    @Positive
import static java.lang.constant.ConstantDescs.BSM_GET_STATIC_FINAL;
    @Positive
import static java.lang.constant.ConstantDescs.CD_Boolean;

    @Positive
@AnnotatedFor({ "interning", "nullness", "value" })
    @Positive
@jdk.internal.ValueBased
    @Positive
public final class Boolean implements java.io.Serializable, Comparable<Boolean>, Constable {

    @Positive
    @Interned
    @Positive
    public static final Boolean TRUE;

    @Positive
    @Interned
    @Positive
    public static final Boolean FALSE;

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public static final Class<Boolean> TYPE;

    @Positive
    @StaticallyExecutable
    @Positive
    @Deprecated()
    @Positive
    @PolyValue
    @Positive
    public Boolean(@PolyValue boolean value) {
    @Positive
    }

    @Positive
    @StaticallyExecutable
    @Positive
    @Deprecated()
    @Positive
    public Boolean(@Nullable String s) {
    @Positive
    }

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public static boolean parseBoolean(@Nullable String s);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    @PolyValue
    @Positive
    public boolean booleanValue(@PolyValue Boolean this);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    @Interned
    @Positive
    @NewObject
    @Positive
    @PolyValue
    @Positive
    public static Boolean valueOf(@PolyValue boolean b);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @Interned
    @Positive
    @NewObject
    @Positive
    @PolyValue
    @Positive
    public static Boolean valueOf(@Nullable @PolyValue String s);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @StringVal({ "true", "false" })
    @Positive
    public static String toString(boolean b);

    @Positive
    @StaticallyExecutable
    @Positive
    @SideEffectFree
    @Positive
    @StringVal({ "true", "false" })
    @Positive
    public String toString();

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int hashCode(boolean value);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public static boolean getBoolean(@Nullable String name);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public int compareTo(Boolean b);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int compare(boolean x, boolean y);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean logicalAnd(boolean a, boolean b);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean logicalOr(boolean a, boolean b);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean logicalXor(boolean a, boolean b);

    @Positive
    @Override
    @Positive
    public Optional<DynamicConstantDesc<Boolean>> describeConstable();
    @Positive
}

// CFWR semantic augmentation - variant 1
