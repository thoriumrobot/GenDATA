/*
    @Positive
 * Copyright (c) 2003, 2021, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.tools.javac.code;

    @Positive
import org.checkerframework.checker.interning.qual.InternedDistinct;
    @Positive
import java.util.Iterator;
    @Positive
import com.sun.tools.javac.tree.JCTree.JCLambda;
    @Positive
import com.sun.tools.javac.util.*;

    @Positive
public class TypeAnnotationPosition {

    @Positive
    public enum TypePathEntryKind {

    @Positive
        ARRAY(0), INNER_TYPE(1), WILDCARD(2), TYPE_ARGUMENT(3);

    @Positive
        public final int tag;
    @Positive
    }

    @Positive
    public static class TypePathEntry {

    @Positive
        public static final int bytesPerEntry;

    @Positive
        public final TypePathEntryKind tag;

    @Positive
        public final int arg;

    @Positive
        @InternedDistinct
    @Positive
        public static final TypePathEntry ARRAY;

    @Positive
        @InternedDistinct
    @Positive
        public static final TypePathEntry INNER_TYPE;

    @Positive
        @InternedDistinct
    @Positive
        public static final TypePathEntry WILDCARD;

    @Positive
        public TypePathEntry(TypePathEntryKind tag, int arg) {
    @Positive
        }

    @Positive
        public static TypePathEntry fromBinary(int tag, int arg);

    @Positive
        @Override
    @Positive
        public String toString();

    @Positive
        @Override
    @Positive
        public boolean equals(Object other);

    @Positive
        @Override
    @Positive
        public int hashCode();
    @Positive
    }

    @Positive
    public static final List<TypePathEntry> emptyPath;

    @Positive
    public final TargetType type;

    @Positive
    public List<TypePathEntry> location;

    @Positive
    public final int pos;

    @Positive
    public boolean isValidOffset;

    @Positive
    public int offset;

    @Positive
    public int[] lvarOffset;

    @Positive
    public int[] lvarLength;

    @Positive
    public int[] lvarIndex;

    @Positive
    public final int bound_index;

    @Positive
    public int parameter_index;

    @Positive
    public final int type_index;

    @Positive
    public final JCLambda onLambda;

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    public boolean emitToClassfile();

    @Positive
    public boolean matchesPos(int pos);

    @Positive
    public void updatePosOffset(int to);

    @Positive
    public boolean hasExceptionIndex();

    @Positive
    public int getExceptionIndex();

    @Positive
    public void setExceptionIndex(final int exception_index);

    @Positive
    public boolean hasCatchType();

    @Positive
    public int getCatchType();

    @Positive
    public int getStartPos();

    @Positive
    public void setCatchInfo(final int catchType, final int startPos);

    @Positive
    public static List<TypePathEntry> getTypePathFromBinary(java.util.List<Integer> list);

    @Positive
    public static List<Integer> getBinaryFromTypePath(java.util.List<TypePathEntry> locs);

    @Positive
    public static TypeAnnotationPosition methodReturn(final List<TypePathEntry> location, final JCLambda onLambda, final int pos);

    @Positive
    public static TypeAnnotationPosition methodReturn(final List<TypePathEntry> location);

    @Positive
    public static TypeAnnotationPosition methodReturn(final int pos);

    @Positive
    public static TypeAnnotationPosition methodReceiver(final List<TypePathEntry> location, final JCLambda onLambda, final int pos);

    @Positive
    public static TypeAnnotationPosition methodReceiver(final List<TypePathEntry> location);

    @Positive
    public static TypeAnnotationPosition methodReceiver(final int pos);

    @Positive
    public static TypeAnnotationPosition methodParameter(final List<TypePathEntry> location, final JCLambda onLambda, final int parameter_index, final int pos);

    @Positive
    public static TypeAnnotationPosition methodParameter(final JCLambda onLambda, final int parameter_index, final int pos);

    @Positive
    public static TypeAnnotationPosition methodParameter(final int parameter_index, final int pos);

    @Positive
    public static TypeAnnotationPosition methodParameter(final List<TypePathEntry> location, final int parameter_index);

    @Positive
    public static TypeAnnotationPosition methodRef(final List<TypePathEntry> location, final JCLambda onLambda, final int pos);

    @Positive
    public static TypeAnnotationPosition methodRef(final List<TypePathEntry> location);

    @Positive
    public static TypeAnnotationPosition constructorRef(final List<TypePathEntry> location, final JCLambda onLambda, final int pos);

    @Positive
    public static TypeAnnotationPosition constructorRef(final List<TypePathEntry> location);

    @Positive
    public static TypeAnnotationPosition field(final List<TypePathEntry> location, final JCLambda onLambda, final int pos);

    @Positive
    public static TypeAnnotationPosition field(final List<TypePathEntry> location);

    @Positive
    public static TypeAnnotationPosition field(final int pos);

    @Positive
    public static TypeAnnotationPosition localVariable(final List<TypePathEntry> location, final JCLambda onLambda, final int pos);

    @Positive
    public static TypeAnnotationPosition localVariable(final JCLambda onLambda, final int pos);

    @Positive
    public static TypeAnnotationPosition localVariable(final List<TypePathEntry> location);

    @Positive
    public static TypeAnnotationPosition exceptionParameter(final List<TypePathEntry> location, final JCLambda onLambda, final int pos);

    @Positive
    public static TypeAnnotationPosition exceptionParameter(final JCLambda onLambda, final int pos);

    @Positive
    public static TypeAnnotationPosition exceptionParameter(final List<TypePathEntry> location);

    @Positive
    public static TypeAnnotationPosition resourceVariable(final List<TypePathEntry> location, final JCLambda onLambda, final int pos);

    @Positive
    public static TypeAnnotationPosition resourceVariable(final JCLambda onLambda, final int pos);

    @Positive
    public static TypeAnnotationPosition resourceVariable(final List<TypePathEntry> location);

    @Positive
    public static TypeAnnotationPosition newObj(final List<TypePathEntry> location, final JCLambda onLambda, final int pos);

    @Positive
    public static TypeAnnotationPosition newObj(final int pos);

    @Positive
    public static TypeAnnotationPosition newObj(final List<TypePathEntry> location);

    @Positive
    public static TypeAnnotationPosition classExtends(final List<TypePathEntry> location, final JCLambda onLambda, final int type_index, final int pos);

    @Positive
    public static TypeAnnotationPosition classExtends(final List<TypePathEntry> location, final JCLambda onLambda, final int pos);

    @Positive
    public static TypeAnnotationPosition classExtends(final List<TypePathEntry> location, final int type_index);

    @Positive
    public static TypeAnnotationPosition classExtends(final int type_index, final int pos);

    @Positive
    public static TypeAnnotationPosition classExtends(final int pos);

    @Positive
    public static TypeAnnotationPosition instanceOf(final List<TypePathEntry> location, final JCLambda onLambda, final int pos);

    @Positive
    public static TypeAnnotationPosition instanceOf(final List<TypePathEntry> location);

    @Positive
    public static TypeAnnotationPosition typeCast(final List<TypePathEntry> location, final JCLambda onLambda, final int type_index, final int pos);

    @Positive
    public static TypeAnnotationPosition typeCast(final List<TypePathEntry> location, final int type_index);

    @Positive
    public static TypeAnnotationPosition methodInvocationTypeArg(final List<TypePathEntry> location, final JCLambda onLambda, final int type_index, final int pos);

    @Positive
    public static TypeAnnotationPosition methodInvocationTypeArg(final List<TypePathEntry> location, final int type_index);

    @Positive
    public static TypeAnnotationPosition constructorInvocationTypeArg(final List<TypePathEntry> location, final JCLambda onLambda, final int type_index, final int pos);

    @Positive
    public static TypeAnnotationPosition constructorInvocationTypeArg(final List<TypePathEntry> location, final int type_index);

    @Positive
    public static TypeAnnotationPosition typeParameter(final List<TypePathEntry> location, final JCLambda onLambda, final int parameter_index, final int pos);

    @Positive
    public static TypeAnnotationPosition typeParameter(final List<TypePathEntry> location, final int parameter_index);

    @Positive
    public static TypeAnnotationPosition methodTypeParameter(final List<TypePathEntry> location, final JCLambda onLambda, final int parameter_index, final int pos);

    @Positive
    public static TypeAnnotationPosition methodTypeParameter(final List<TypePathEntry> location, final int parameter_index);

    @Positive
    public static TypeAnnotationPosition methodThrows(final List<TypePathEntry> location, final JCLambda onLambda, final int type_index, final int pos);

    @Positive
    public static TypeAnnotationPosition methodThrows(final List<TypePathEntry> location, final int type_index);

    @Positive
    public static TypeAnnotationPosition methodRefTypeArg(final List<TypePathEntry> location, final JCLambda onLambda, final int type_index, final int pos);

    @Positive
    public static TypeAnnotationPosition methodRefTypeArg(final List<TypePathEntry> location, final int type_index);

    @Positive
    public static TypeAnnotationPosition constructorRefTypeArg(final List<TypePathEntry> location, final JCLambda onLambda, final int type_index, final int pos);

    @Positive
    public static TypeAnnotationPosition constructorRefTypeArg(final List<TypePathEntry> location, final int type_index);

    @Positive
    public static TypeAnnotationPosition typeParameterBound(final List<TypePathEntry> location, final JCLambda onLambda, final int parameter_index, final int bound_index, final int pos);

    @Positive
    public static TypeAnnotationPosition typeParameterBound(final List<TypePathEntry> location, final int parameter_index, final int bound_index);

    @Positive
    public static TypeAnnotationPosition methodTypeParameterBound(final List<TypePathEntry> location, final JCLambda onLambda, final int parameter_index, final int bound_index, final int pos);

    @Positive
    public static TypeAnnotationPosition methodTypeParameterBound(final List<TypePathEntry> location, final int parameter_index, final int bound_index);

    @Positive
    public static final TypeAnnotationPosition unknown;
    @Positive
}

// CFWR semantic augmentation - variant 1
