/*
 * Copyright (c) 2003, 2021, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.tools.javac.code;

import org.checkerframework.checker.interning.qual.InternedDistinct;
import java.util.Iterator;
import com.sun.tools.javac.tree.JCTree.JCLambda;
import com.sun.tools.javac.util.*;

public class TypeAnnotationPosition {

    public enum TypePathEntryKind {

        ARRAY(0), INNER_TYPE(1), WILDCARD(2), TYPE_ARGUMENT(3);

        public final int tag;
    }

    public static class TypePathEntry {

        public static final int bytesPerEntry;

        public final TypePathEntryKind tag;

        public final int arg;

        @InternedDistinct
        public static final TypePathEntry ARRAY;

        @InternedDistinct
        public static final TypePathEntry INNER_TYPE;

        @InternedDistinct
        public static final TypePathEntry WILDCARD;

        public TypePathEntry(TypePathEntryKind tag, int arg) {
        }

        public static TypePathEntry fromBinary(int tag, int arg);

        @Override
        public String toString();

        @Override
        public boolean equals(Object other);

        @Override
        public int hashCode();
    }

    public static final List<TypePathEntry> emptyPath;

    public final TargetType type;

    public List<TypePathEntry> location;

    public final int pos;

    public boolean isValidOffset;

    public int offset;

    public int[] lvarOffset;

    public int[] lvarLength;

    public int[] lvarIndex;

    public final int bound_index;

    public int parameter_index;

    public final int type_index;

    public final JCLambda onLambda;

    @Override
    public String toString();

    public boolean emitToClassfile();

    public boolean matchesPos(int pos);

    public void updatePosOffset(int to);

    public boolean hasExceptionIndex();

    public int getExceptionIndex();

    public void setExceptionIndex(final int exception_index);

    public boolean hasCatchType();

    public int getCatchType();

    public int getStartPos();

    public void setCatchInfo(final int catchType, final int startPos);

    public static List<TypePathEntry> getTypePathFromBinary(java.util.List<Integer> list);

    public static List<Integer> getBinaryFromTypePath(java.util.List<TypePathEntry> locs);

    public static TypeAnnotationPosition methodReturn(final List<TypePathEntry> location, final JCLambda onLambda, final int pos);

    public static TypeAnnotationPosition methodReturn(final List<TypePathEntry> location);

    public static TypeAnnotationPosition methodReturn(final int pos);

    public static TypeAnnotationPosition methodReceiver(final List<TypePathEntry> location, final JCLambda onLambda, final int pos);

    public static TypeAnnotationPosition methodReceiver(final List<TypePathEntry> location);

    public static TypeAnnotationPosition methodReceiver(final int pos);

    public static TypeAnnotationPosition methodParameter(final List<TypePathEntry> location, final JCLambda onLambda, final int parameter_index, final int pos);

    public static TypeAnnotationPosition methodParameter(final JCLambda onLambda, final int parameter_index, final int pos);

    public static TypeAnnotationPosition methodParameter(final int parameter_index, final int pos);

    public static TypeAnnotationPosition methodParameter(final List<TypePathEntry> location, final int parameter_index);

    public static TypeAnnotationPosition methodRef(final List<TypePathEntry> location, final JCLambda onLambda, final int pos);

    public static TypeAnnotationPosition methodRef(final List<TypePathEntry> location);

    public static TypeAnnotationPosition constructorRef(final List<TypePathEntry> location, final JCLambda onLambda, final int pos);

    public static TypeAnnotationPosition constructorRef(final List<TypePathEntry> location);

    public static TypeAnnotationPosition field(final List<TypePathEntry> location, final JCLambda onLambda, final int pos);

    public static TypeAnnotationPosition field(final List<TypePathEntry> location);

    public static TypeAnnotationPosition field(final int pos);

    public static TypeAnnotationPosition localVariable(final List<TypePathEntry> location, final JCLambda onLambda, final int pos);

    public static TypeAnnotationPosition localVariable(final JCLambda onLambda, final int pos);

    public static TypeAnnotationPosition localVariable(final List<TypePathEntry> location);

    public static TypeAnnotationPosition exceptionParameter(final List<TypePathEntry> location, final JCLambda onLambda, final int pos);

    public static TypeAnnotationPosition exceptionParameter(final JCLambda onLambda, final int pos);

    public static TypeAnnotationPosition exceptionParameter(final List<TypePathEntry> location);

    public static TypeAnnotationPosition resourceVariable(final List<TypePathEntry> location, final JCLambda onLambda, final int pos);

    public static TypeAnnotationPosition resourceVariable(final JCLambda onLambda, final int pos);

    public static TypeAnnotationPosition resourceVariable(final List<TypePathEntry> location);

    public static TypeAnnotationPosition newObj(final List<TypePathEntry> location, final JCLambda onLambda, final int pos);

    public static TypeAnnotationPosition newObj(final int pos);

    public static TypeAnnotationPosition newObj(final List<TypePathEntry> location);

    public static TypeAnnotationPosition classExtends(final List<TypePathEntry> location, final JCLambda onLambda, final int type_index, final int pos);

    public static TypeAnnotationPosition classExtends(final List<TypePathEntry> location, final JCLambda onLambda, final int pos);

    public static TypeAnnotationPosition classExtends(final List<TypePathEntry> location, final int type_index);

    public static TypeAnnotationPosition classExtends(final int type_index, final int pos);

    public static TypeAnnotationPosition classExtends(final int pos);

    public static TypeAnnotationPosition instanceOf(final List<TypePathEntry> location, final JCLambda onLambda, final int pos);

    public static TypeAnnotationPosition instanceOf(final List<TypePathEntry> location);

    public static TypeAnnotationPosition typeCast(final List<TypePathEntry> location, final JCLambda onLambda, final int type_index, final int pos);

    public static TypeAnnotationPosition typeCast(final List<TypePathEntry> location, final int type_index);

    public static TypeAnnotationPosition methodInvocationTypeArg(final List<TypePathEntry> location, final JCLambda onLambda, final int type_index, final int pos);

    public static TypeAnnotationPosition methodInvocationTypeArg(final List<TypePathEntry> location, final int type_index);

    public static TypeAnnotationPosition constructorInvocationTypeArg(final List<TypePathEntry> location, final JCLambda onLambda, final int type_index, final int pos);

    public static TypeAnnotationPosition constructorInvocationTypeArg(final List<TypePathEntry> location, final int type_index);

    public static TypeAnnotationPosition typeParameter(final List<TypePathEntry> location, final JCLambda onLambda, final int parameter_index, final int pos);

    public static TypeAnnotationPosition typeParameter(final List<TypePathEntry> location, final int parameter_index);

    public static TypeAnnotationPosition methodTypeParameter(final List<TypePathEntry> location, final JCLambda onLambda, final int parameter_index, final int pos);

    public static TypeAnnotationPosition methodTypeParameter(final List<TypePathEntry> location, final int parameter_index);

    public static TypeAnnotationPosition methodThrows(final List<TypePathEntry> location, final JCLambda onLambda, final int type_index, final int pos);

    public static TypeAnnotationPosition methodThrows(final List<TypePathEntry> location, final int type_index);

    public static TypeAnnotationPosition methodRefTypeArg(final List<TypePathEntry> location, final JCLambda onLambda, final int type_index, final int pos);

    public static TypeAnnotationPosition methodRefTypeArg(final List<TypePathEntry> location, final int type_index);

    public static TypeAnnotationPosition constructorRefTypeArg(final List<TypePathEntry> location, final JCLambda onLambda, final int type_index, final int pos);

    public static TypeAnnotationPosition constructorRefTypeArg(final List<TypePathEntry> location, final int type_index);

    public static TypeAnnotationPosition typeParameterBound(final List<TypePathEntry> location, final JCLambda onLambda, final int parameter_index, final int bound_index, final int pos);

    public static TypeAnnotationPosition typeParameterBound(final List<TypePathEntry> location, final int parameter_index, final int bound_index);

    public static TypeAnnotationPosition methodTypeParameterBound(final List<TypePathEntry> location, final JCLambda onLambda, final int parameter_index, final int bound_index, final int pos);

    public static TypeAnnotationPosition methodTypeParameterBound(final List<TypePathEntry> location, final int parameter_index, final int bound_index);

    public static final TypeAnnotationPosition unknown;
}
