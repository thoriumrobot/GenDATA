/*
    @Positive
 * Copyright (c) 1994, 2021, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.common.value.qual.PolyValue;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;

    @Positive
@AnnotatedFor({ "lock", "nullness", "value" })
    @Positive
public abstract class Number implements java.io.Serializable {

    @Positive
    public Number() {
    @Positive
    }

    @Positive
    @Pure
    @Positive
    @PolyValue
    @Positive
    public abstract int intValue(@GuardSatisfied @PolyValue Number this);

    @Positive
    @Pure
    @Positive
    @PolyValue
    @Positive
    public abstract long longValue(@GuardSatisfied @PolyValue Number this);

    @Positive
    @Pure
    @Positive
    @PolyValue
    @Positive
    public abstract float floatValue(@GuardSatisfied @PolyValue Number this);

    @Positive
    @Pure
    @Positive
    @PolyValue
    @Positive
    public abstract double doubleValue(@GuardSatisfied @PolyValue Number this);

    @Positive
    @Pure
    @Positive
    @PolyValue
    @Positive
    public byte byteValue(@GuardSatisfied @PolyValue Number this);

    @Positive
    @Pure
    @Positive
    @PolyValue
    @Positive
    public short shortValue(@GuardSatisfied @PolyValue Number this);
    @Positive
}

// CFWR semantic augmentation - variant 1
