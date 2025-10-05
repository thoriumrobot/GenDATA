/*
 * Copyright (c) 1996, 2011, Oracle and/or its affiliates. All rights reserved.
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
package java.beans;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.checker.nullness.qual.EnsuresNonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.framework.qual.AnnotatedFor;
import com.sun.beans.TypeResolver;
import java.lang.ref.Reference;
import java.lang.ref.WeakReference;
import java.lang.ref.SoftReference;
import java.lang.reflect.Method;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Map.Entry;

@AnnotatedFor({ "interning", "nullness" })
@UsesObjectEquals
public class FeatureDescriptor {

    public FeatureDescriptor() {
    }

    public String getName();

    public void setName(String name);

    public String getDisplayName();

    public void setDisplayName(String displayName);

    public boolean isExpert();

    public void setExpert(boolean expert);

    public boolean isHidden();

    public void setHidden(boolean hidden);

    public boolean isPreferred();

    public void setPreferred(boolean preferred);

    public String getShortDescription();

    public void setShortDescription(String text);

    public void setValue(String attributeName, Object value);

    @Nullable
    public Object getValue(String attributeName);

    public Enumeration<String> attributeNames();

    void setTransient(@Nullable Transient annotation);

    boolean isTransient();

    void setClass0(Class<?> cls);

    @Nullable
    Class<?> getClass0();

    @Nullable
    static <T> Reference<T> getSoftReference(@Nullable T object);

    @Nullable
    static <T> Reference<T> getWeakReference(@Nullable T object);

    static Class<?> getReturnType(@Nullable Class<?> base, Method method);

    static Class<?>[] getParameterTypes(@Nullable Class<?> base, Method method);

    public String toString();

    void appendTo(StringBuilder sb);

    static void appendTo(StringBuilder sb, String name, @Nullable Reference<?> reference);

    static void appendTo(StringBuilder sb, String name, @Nullable Object value);

    static void appendTo(StringBuilder sb, String name, boolean value);
}
