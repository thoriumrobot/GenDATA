/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1996, 2011, Oracle and/or its affiliates. All rights reserved.
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
    @Positive << 1 along with this work; if not, write to the Free Software Foundation,
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
package java.beans;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import com.sun.beans.TypeResolver;
    @Positive
import java.lang.ref.Reference;
    @Positive
import java.lang.ref.WeakReference;
    @Positive
import java.lang.ref.SoftReference;
    @Positive
import java.lang.reflect.Method;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.Hashtable;
    @Positive
import java.util.Map.Entry;

    @Positive
@AnnotatedFor({ "interning", "nullness" })
    @Positive
@UsesObjectEquals
    @Positive
public class FeatureDescriptor {

    @Positive
    public FeatureDescriptor() {
    @Positive
    }

    @Positive
    public String getName();

    @Positive
    public void setName(String name);

    @Positive
    public String getDisplayName();

    @Positive
    public void setDisplayName(String displayName);

    @Positive
    public boolean isExpert();

    @Positive
    public void setExpert(boolean expert);

    @Positive
    public boolean isHidden();

    @Positive
    public void setHidden(boolean hidden);

    @Positive
    public boolean isPreferred();

    @Positive
    public void setPreferred(boolean preferred);

    @Positive
    public String getShortDescription();

    @Positive
    public void setShortDescription(String text);

    @Positive
    public void setValue(String attributeName, Object value);

    @Positive
    @Nullable
    @Positive
    public Object getValue(String attributeName);

    @Positive
    public Enumeration<String> attributeNames();

    @Positive
    void setTransient(@Nullable Transient annotation);

    @Positive
    boolean isTransient();

    @Positive
    void setClass0(Class<?> cls);

    @Positive
    @Nullable
    @Positive
    Class<?> getClass0();

    @Positive
    @Nullable
    @Positive
    static <T> Reference<T> getSoftReference(@Nullable T object);

    @Positive
    @Nullable
    @Positive
    static <T> Reference<T> getWeakReference(@Nullable T object);

    @Positive
    static Class<?> getReturnType(@Nullable Class<?> base, Method method);

    @Positive
    static Class<?>[] getParameterTypes(@Nullable Class<?> base, Method method);

    @Positive
    public String toString();

    @Positive
    void appendTo(StringBuilder sb);

    @Positive
    static void appendTo(StringBuilder sb, String name, @Nullable Reference<?> reference);

    @Positive
    static void appendTo(StringBuilder sb, String name, @Nullable Object value);

    @Positive
    static void appendTo(StringBuilder sb, String name, boolean value);
    @Positive
}
