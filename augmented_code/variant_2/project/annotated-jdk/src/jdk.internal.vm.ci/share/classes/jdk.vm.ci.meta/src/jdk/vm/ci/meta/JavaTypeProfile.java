/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
package jdk.vm.ci.meta;

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
import java.lang.reflect.Modifier;
    @Positive
import java.util.ArrayList;
    @Positive
import jdk.vm.ci.meta.JavaTypeProfile.ProfiledType;

    @Positive
public final class JavaTypeProfile extends AbstractJavaProfile<ProfiledType, ResolvedJavaType> {

    @Positive
    public JavaTypeProfile(TriState nullSeen, double notRecordedProbability, ProfiledType[] pitems) {
    @Positive
    }

    @Positive
    public TriState getNullSeen();

    @Positive
    public ProfiledType[] getTypes();

    @Positive
    public JavaTypeProfile restrict(JavaTypeProfile otherProfile);

    @Positive
    public JavaTypeProfile restrict(ResolvedJavaType declaredType, boolean nonNull);

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object other);

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    public static class ProfiledType extends AbstractProfiledItem<ResolvedJavaType> {

    @Positive
        public ProfiledType(ResolvedJavaType type, double probability) {
    @Positive
        }

    @Positive
        public ResolvedJavaType getType();

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    public boolean allTypesRecorded();

    @Positive
    public ResolvedJavaType asSingleType();
    @Positive
}
