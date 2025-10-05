/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2012, 2021, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
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
package java.lang.reflect;

    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.lang.annotation.*;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Map;
    @Positive
import java.util.Objects;
    @Positive
import java.util.StringJoiner;
    @Positive
import java.util.stream.Stream;
    @Positive
import java.util.stream.Collectors;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import sun.reflect.annotation.AnnotationParser;
    @Positive
import sun.reflect.annotation.AnnotationSupport;
    @Positive
import sun.reflect.annotation.TypeAnnotationParser;
    @Positive
import sun.reflect.annotation.TypeAnnotation;
    @Positive
import sun.reflect.generics.reflectiveObjects.ParameterizedTypeImpl;
    @Positive
import sun.reflect.generics.repository.ConstructorRepository;

    @Positive
@AnnotatedFor({ "nullness" })
    @Positive
public abstract sealed class Executable extends AccessibleObject implements Member, GenericDeclaration permits Constructor, Method {

    @Positive
    abstract byte[] getAnnotationBytes();

    @Positive
    abstract boolean hasGenericInformation();

    @Positive
    abstract ConstructorRepository getGenericInfo();

    @Positive
    boolean equalParamTypes(Class<?>[] params1, Class<?>[] params2);

    @Positive
    Annotation[][] parseParameterAnnotations(byte[] parameterAnnotations);

    @Positive
    void printModifiersIfNonzero(StringBuilder sb, int mask, boolean isDefault);

    @Positive
    String sharedToString(int modifierMask, boolean isDefault, Class<?>[] parameterTypes, Class<?>[] exceptionTypes);

    @Positive
    abstract void specificToStringHeader(StringBuilder sb);

    @Positive
    static String typeVarBounds(TypeVariable<?> typeVar);

    @Positive
    String sharedToGenericString(int modifierMask, boolean isDefault);

    @Positive
    abstract void specificToGenericStringHeader(StringBuilder sb);

    @Positive
    public abstract Class<?> getDeclaringClass();

    @Positive
    public abstract String getName();

    @Positive
    public abstract int getModifiers();

    @Positive
    public abstract TypeVariable<?>[] getTypeParameters();

    @Positive
    abstract Class<?>[] getSharedParameterTypes();

    @Positive
    abstract Class<?>[] getSharedExceptionTypes();

    @Positive
    public abstract Class<?>[] getParameterTypes();

    @Positive
    public int getParameterCount();

    @Positive
    public Type[] getGenericParameterTypes();

    @Positive
    Type[] getAllGenericParameterTypes();

    @Positive
    public Parameter[] getParameters();

    @Positive
    boolean hasRealParameterData();

    @Positive
    native byte[] getTypeAnnotationBytes0();

    @Positive
    byte[] getTypeAnnotationBytes();

    @Positive
    public abstract Class<?>[] getExceptionTypes();

    @Positive
    public Type[] getGenericExceptionTypes();

    @Positive
    public abstract String toGenericString();

    @Positive
    public boolean isVarArgs();

    @Positive
    public boolean isSynthetic();

    @Positive
    public abstract Annotation[][] getParameterAnnotations();

    @Positive
    Annotation[][] sharedGetParameterAnnotations(Class<?>[] parameterTypes, byte[] parameterAnnotations);

    @Positive
    abstract boolean handleParameterNumberMismatch(int resultLength, Class<?>[] parameterTypes);

    @Positive
    @Override
    @Positive
    @Nullable
    @Positive
    public <T extends Annotation> T getAnnotation(Class<T> annotationClass);

    @Positive
    @Override
    @Positive
    public <T extends Annotation> T[] getAnnotationsByType(Class<T> annotationClass);

    @Positive
    @Override
    @Positive
    public Annotation[] getDeclaredAnnotations();

    @Positive
    public abstract AnnotatedType getAnnotatedReturnType();

    @Positive
    AnnotatedType getAnnotatedReturnType0(Type returnType);

    @Positive
    @Nullable
    @Positive
    public AnnotatedType getAnnotatedReceiverType();

    @Positive
    Type parameterize(Class<?> c);

    @Positive
    public AnnotatedType[] getAnnotatedParameterTypes();

    @Positive
    public AnnotatedType[] getAnnotatedExceptionTypes();
    @Positive
}
