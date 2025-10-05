/*
 * Copyright (c) 2000, 2013, Oracle and/or its affiliates. All rights reserved.
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
package javax.management.openmbean;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.io.InvalidObjectException;
import java.io.ObjectStreamException;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.Date;
import java.util.Map;
import java.util.HashMap;
import javax.management.ObjectName;

public final class SimpleType<T> extends OpenType<T> {

    public static final SimpleType<Void> VOID;

    public static final SimpleType<Boolean> BOOLEAN;

    public static final SimpleType<Character> CHARACTER;

    public static final SimpleType<Byte> BYTE;

    public static final SimpleType<Short> SHORT;

    public static final SimpleType<Integer> INTEGER;

    public static final SimpleType<Long> LONG;

    public static final SimpleType<Float> FLOAT;

    public static final SimpleType<Double> DOUBLE;

    public static final SimpleType<String> STRING;

    public static final SimpleType<BigDecimal> BIGDECIMAL;

    public static final SimpleType<BigInteger> BIGINTEGER;

    public static final SimpleType<Date> DATE;

    public static final SimpleType<ObjectName> OBJECTNAME;

    public boolean isValue(Object obj);

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public int hashCode();

    public String toString();

    public Object readResolve() throws ObjectStreamException;
}
