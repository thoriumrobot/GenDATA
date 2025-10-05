/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
public class HotSpotResolvedJavaFieldImpl {
/*
    @Positive
 * Copyright (c) 2011, 2019, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.
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
package jdk.vm.ci.hotspot;

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
import static jdk.vm.ci.hotspot.HotSpotJVMCIRuntime.runtime;
    @Positive
import static jdk.vm.ci.hotspot.HotSpotVMConfig.config;
    @Positive
import static jdk.vm.ci.hotspot.UnsafeAccess.UNSAFE;
    @Positive
import static jdk.internal.misc.Unsafe.ADDRESS_SIZE;
    @Positive
import java.lang.annotation.Annotation;
    @Positive
import jdk.internal.vm.annotation.Stable;
    @Positive
import jdk.vm.ci.meta.JavaConstant;
    @Positive
import jdk.vm.ci.meta.JavaType;
    @Positive
import jdk.vm.ci.meta.ResolvedJavaType;
    @Positive
import jdk.vm.ci.meta.UnresolvedJavaType;

    @Positive
class HotSpotResolvedJavaFieldImpl implements HotSpotResolvedJavaField {

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Override
    @Positive
    public int getModifiers();

    @Positive
    @Override
    @Positive
    public boolean isInternal();

    @Positive
    @Override
    @Positive
    public boolean isInObject(JavaConstant object);

    @Positive
    @Override
    @Positive
    public HotSpotResolvedObjectTypeImpl getDeclaringClass();

    @Positive
    @Override
    @Positive
    public String getName();

    @Positive
    @Override
    @Positive
    public JavaType getType();

    @Positive
    @Override
    @Positive
    public int getOffset();

    @Positive
    int getIndex();

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    @Override
    @Positive
    public boolean isSynthetic();

    @Positive
    @Override
    @Positive
    public boolean isStable();

    @Positive
    @Override
    @Positive
    public Annotation[] getAnnotations();

    @Positive
    @Override
    @Positive
    public Annotation[] getDeclaredAnnotations();

    @Positive
    @Override
    @Positive
    public <T extends Annotation> T getAnnotation(Class<T> annotationClass);
    @Positive
}

}