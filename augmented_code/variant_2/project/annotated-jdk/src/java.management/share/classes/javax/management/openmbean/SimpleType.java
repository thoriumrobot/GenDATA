/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2000, 2013, Oracle and/or its affiliates. All rights reserved.
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
package javax.management.openmbean;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import java.io.InvalidObjectException;
    @Positive
import java.io.ObjectStreamException;
    @Positive
import java.math.BigDecimal;
    @Positive
import java.math.BigInteger;
    @Positive
import java.util.Date;
    @Positive
import java.util.Map;
    @Positive
import java.util.HashMap;
    @Positive
import javax.management.ObjectName;

    @Positive
public final class SimpleType<T> extends OpenType<T> {

    @Positive
    public static final SimpleType<Void> VOID;

    @Positive
    public static final SimpleType<Boolean> BOOLEAN;

    @Positive
    public static final SimpleType<Character> CHARACTER;

    @Positive
    public static final SimpleType<Byte> BYTE;

    @Positive
    public static final SimpleType<Short> SHORT;

    @Positive
    public static final SimpleType<Integer> INTEGER;

    @Positive
    public static final SimpleType<Long> LONG;

    @Positive
    public static final SimpleType<Float> FLOAT;

    @Positive
    public static final SimpleType<Double> DOUBLE;

    @Positive
    public static final SimpleType<String> STRING;

    @Positive
    public static final SimpleType<BigDecimal> BIGDECIMAL;

    @Positive
    public static final SimpleType<BigInteger> BIGINTEGER;

    @Positive
    public static final SimpleType<Date> DATE;

    @Positive
    public static final SimpleType<ObjectName> OBJECTNAME;

    @Positive
    public boolean isValue(Object obj);

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int hashCode();

    @Positive
    public String toString();

    @Positive
    public Object readResolve() throws ObjectStreamException;
    @Positive
}
