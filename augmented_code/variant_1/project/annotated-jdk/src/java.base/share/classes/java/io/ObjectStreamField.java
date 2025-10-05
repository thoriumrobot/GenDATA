/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.io;

    @Positive
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.lang.reflect.Field;
    @Positive
import jdk.internal.reflect.CallerSensitive;
    @Positive
import jdk.internal.reflect.Reflection;
    @Positive
import sun.reflect.misc.ReflectUtil;

    @Positive
@AnnotatedFor({ "interning", "lock", "nullness" })
    @Positive
@UsesObjectEquals
    @Positive
public class ObjectStreamField implements Comparable<Object> {

    @Positive
    public ObjectStreamField(String name, Class<?> type) {
    @Positive
    }

    @Positive
    public ObjectStreamField(String name, Class<?> type, boolean unshared) {
    @Positive
    }

    @Positive
    static String getClassSignature(Class<?> cl);

    @Positive
    static StringBuilder appendClassSignature(StringBuilder sbuf, Class<?> cl);

    @Positive
    public String getName();

    @Positive
    @SuppressWarnings("removal")
    @Positive
    @CallerSensitive
    @Positive
    public Class<?> getType();

    @Positive
    public char getTypeCode();

    @Positive
    @Nullable
    @Positive
    @Interned
    @Positive
    public String getTypeString();

    @Positive
    public int getOffset();

    @Positive
    protected void setOffset(int offset);

    @Positive
    @Pure
    @Positive
    public boolean isPrimitive(@GuardSatisfied ObjectStreamField this);

    @Positive
    @Pure
    @Positive
    public boolean isUnshared(@GuardSatisfied ObjectStreamField this);

    @Positive
    @Pure
    @Positive
    public int compareTo(@GuardSatisfied ObjectStreamField this, @GuardSatisfied Object obj);

    @Positive
    @SideEffectFree
    @Positive
    public String toString(@GuardSatisfied ObjectStreamField this);

    @Positive
    Field getField();

    @Positive
    String getSignature();
    @Positive
}
