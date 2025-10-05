/*
 * Copyright (c) 2000, 2014, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.framework.qual.AnnotatedFor;
import org.checkerframework.checker.nullness.qual.EnsuresKeyForIf;
import org.checkerframework.checker.nullness.qual.MonotonicNonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import java.util.Set;
import java.util.TreeMap;
import java.util.Collections;
import java.util.Iterator;

@AnnotatedFor("nullness")
public class CompositeType extends OpenType<CompositeData> {

    public CompositeType(String typeName, String description, String[] itemNames, String[] itemDescriptions, OpenType<?>[] itemTypes) throws OpenDataException {
    }

    @EnsuresKeyForIf(expression = { "#1" }, result = true, map = { "this" })
    @Pure
    public boolean containsKey(@Nullable String itemName);

    @Nullable
    public String getDescription(@Nullable String itemName);

    @Nullable
    public OpenType<?> getType(@Nullable String itemName);

    public Set<String> keySet();

    public boolean isValue(@Nullable Object obj);

    @Override
    boolean isAssignableFrom(OpenType<?> ot);

    public boolean equals(@Nullable Object obj);

    public int hashCode();

    public String toString();
}
