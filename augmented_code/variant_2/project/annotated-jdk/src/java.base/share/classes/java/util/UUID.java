/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2003, 2020, Oracle and/or its affiliates. All rights reserved.
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
import java.security.*;
    @Positive
import jdk.internal.access.JavaLangAccess;
    @Positive
import jdk.internal.access.SharedSecrets;

    @Positive
@AnnotatedFor({ "lock", "nullness", "index" })
    @Positive
public final class UUID implements java.io.Serializable, Comparable<UUID> {

    @Positive
    private static class Holder {
    @Positive
    }

    @Positive
    public UUID(long mostSigBits, long leastSigBits) {
    @Positive
    }

    @Positive
    public static UUID randomUUID();

    @Positive
    public static UUID nameUUIDFromBytes(byte[] name);

    @Positive
    public static UUID fromString(String name);

    @Positive
    public long getLeastSignificantBits(@GuardSatisfied UUID this);

    @Positive
    public long getMostSignificantBits(@GuardSatisfied UUID this);

    @Positive
    public int version();

    @Positive
    public int variant();

    @Positive
    public long timestamp();

    @Positive
    public int clockSequence();

    @Positive
    public long node();

    @Positive
    @SideEffectFree
    @Positive
    @Override
    @Positive
    public String toString(@GuardSatisfied UUID this);

    @Positive
    @Pure
    @Positive
    @Override
    @Positive
    public int hashCode(@GuardSatisfied UUID this);

    @Positive
    @Pure
    @Positive
    @Override
    @Positive
    public boolean equals(@GuardSatisfied UUID this, @GuardSatisfied @Nullable Object obj);

    @Positive
    @Pure
    @Positive
    @Override
    @Positive
    public int compareTo(@GuardSatisfied UUID this, @GuardSatisfied UUID val);
    @Positive
}
