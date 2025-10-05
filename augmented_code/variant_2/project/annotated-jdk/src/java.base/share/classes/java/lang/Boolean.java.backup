/*
 * Copyright (c) 1994, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.lang;

import org.checkerframework.checker.interning.qual.Interned;
import org.checkerframework.checker.lock.qual.NewObject;
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.common.value.qual.PolyValue;
import org.checkerframework.common.value.qual.StaticallyExecutable;
import org.checkerframework.common.value.qual.StringVal;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import jdk.internal.vm.annotation.IntrinsicCandidate;
import java.lang.constant.Constable;
import java.lang.constant.ConstantDesc;
import java.lang.constant.ConstantDescs;
import java.lang.constant.DynamicConstantDesc;
import java.util.Optional;
import static java.lang.constant.ConstantDescs.BSM_GET_STATIC_FINAL;
import static java.lang.constant.ConstantDescs.CD_Boolean;

@AnnotatedFor({ "interning", "nullness", "value" })
@jdk.internal.ValueBased
public final class Boolean implements java.io.Serializable, Comparable<Boolean>, Constable {

    @Interned
    public static final Boolean TRUE;

    @Interned
    public static final Boolean FALSE;

    @SuppressWarnings("unchecked")
    public static final Class<Boolean> TYPE;

    @StaticallyExecutable
    @Deprecated()
    @PolyValue
    public Boolean(@PolyValue boolean value) {
    }

    @StaticallyExecutable
    @Deprecated()
    public Boolean(@Nullable String s) {
    }

    @Pure
    @StaticallyExecutable
    @EnsuresNonNullIf(expression = "#1", result = true)
    public static boolean parseBoolean(@Nullable String s);

    @Pure
    @StaticallyExecutable
    @IntrinsicCandidate
    @PolyValue
    public boolean booleanValue(@PolyValue Boolean this);

    @Pure
    @StaticallyExecutable
    @IntrinsicCandidate
    @Interned
    @NewObject
    @PolyValue
    public static Boolean valueOf(@PolyValue boolean b);

    @Pure
    @StaticallyExecutable
    @Interned
    @NewObject
    @PolyValue
    public static Boolean valueOf(@Nullable @PolyValue String s);

    @Pure
    @StaticallyExecutable
    @StringVal({ "true", "false" })
    public static String toString(boolean b);

    @StaticallyExecutable
    @SideEffectFree
    @StringVal({ "true", "false" })
    public String toString();

    @Pure
    @StaticallyExecutable
    @Override
    public int hashCode();

    @Pure
    @StaticallyExecutable
    public static int hashCode(boolean value);

    @Pure
    @StaticallyExecutable
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    @Pure
    @StaticallyExecutable
    @EnsuresNonNullIf(expression = "#1", result = true)
    public static boolean getBoolean(@Nullable String name);

    @Pure
    @StaticallyExecutable
    public int compareTo(Boolean b);

    @Pure
    @StaticallyExecutable
    public static int compare(boolean x, boolean y);

    @Pure
    @StaticallyExecutable
    public static boolean logicalAnd(boolean a, boolean b);

    @Pure
    @StaticallyExecutable
    public static boolean logicalOr(boolean a, boolean b);

    @Pure
    @StaticallyExecutable
    public static boolean logicalXor(boolean a, boolean b);

    @Override
    public Optional<DynamicConstantDesc<Boolean>> describeConstable();
}
