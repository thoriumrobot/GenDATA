/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2011, 2019, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
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
import java.util.ArrayList;
    @Positive
import java.util.List;
    @Positive
import jdk.vm.ci.meta.JavaKind;
    @Positive
import jdk.vm.ci.meta.JavaType;
    @Positive
import jdk.vm.ci.meta.ResolvedJavaType;
    @Positive
import jdk.vm.ci.meta.Signature;
    @Positive
import jdk.vm.ci.meta.UnresolvedJavaType;

    @Positive
public class HotSpotSignature implements Signature {

    @Positive
    public HotSpotSignature(HotSpotJVMCIRuntime runtime, String signature) {
    @Positive
    }

    @Positive
    public HotSpotSignature(HotSpotJVMCIRuntime runtime, ResolvedJavaType returnType, ResolvedJavaType... parameterTypes) {
    @Positive
    }

    @Positive
    @Override
    @Positive
    public int getParameterCount(boolean withReceiver);

    @Positive
    @Override
    @Positive
    public JavaKind getParameterKind(int index);

    @Positive
    @Override
    @Positive
    public JavaType getParameterType(int index, ResolvedJavaType accessingClass);

    @Positive
    @Override
    @Positive
    public String toMethodDescriptor();

    @Positive
    @Override
    @Positive
    public JavaKind getReturnKind();

    @Positive
    @Override
    @Positive
    public JavaType getReturnType(ResolvedJavaType accessingClass);

    @Positive
    @Override
    @Positive
    public String toString();

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
}
