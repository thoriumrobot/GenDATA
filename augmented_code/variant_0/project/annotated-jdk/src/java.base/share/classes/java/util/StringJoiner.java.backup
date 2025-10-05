/*
 * Copyright (c) 2013, 2021, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.lock.qual.ReleasesNoLocks;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import jdk.internal.access.JavaLangAccess;
import jdk.internal.access.SharedSecrets;

@AnnotatedFor({ "nullness" })
public final class StringJoiner {

    @SideEffectFree
    public StringJoiner(CharSequence delimiter) {
    }

    @SideEffectFree
    public StringJoiner(CharSequence delimiter, CharSequence prefix, CharSequence suffix) {
    }

    public StringJoiner setEmptyValue(CharSequence emptyValue);

    @SideEffectFree
    @Override
    public String toString();

    @ReleasesNoLocks
    public StringJoiner add(@Nullable CharSequence newElement);

    @ReleasesNoLocks
    public StringJoiner merge(StringJoiner other);

    @Pure
    public int length();
}
