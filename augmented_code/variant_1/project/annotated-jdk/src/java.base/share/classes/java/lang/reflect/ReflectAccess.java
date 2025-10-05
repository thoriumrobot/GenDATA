/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
public class ReflectAccess {
/*
    @Copyright * Positive (c) 2001, 2020, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import jdk.internal.reflect.MethodAccessor;
    @Positive
import jdk.internal.reflect.ConstructorAccessor;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
class ReflectAccess implements jdk.internal.access.JavaLangReflectAccess {

    @Positive
    public <T> Constructor<T> newConstructor(Class<T> declaringClass, Class<?>[] parameterTypes, Class<?>[] checkedExceptions, int modifiers, int slot, String signature, byte[] annotations, byte[] parameterAnnotations);

    @Positive
    public MethodAccessor getMethodAccessor(Method m);

    @Positive
    public void setMethodAccessor(Method m, MethodAccessor accessor);

    @Positive
    public ConstructorAccessor getConstructorAccessor(Constructor<?> c);

    @Positive
    public void setConstructorAccessor(Constructor<?> c, ConstructorAccessor accessor);

    @Positive
    public int getConstructorSlot(Constructor<?> c);

    @Positive
    public String getConstructorSignature(Constructor<?> c);

    @Positive
    public byte[] getConstructorAnnotations(Constructor<?> c);

    @Positive
    public byte[] getConstructorParameterAnnotations(Constructor<?> c);

    @Positive
    public byte[] getExecutableTypeAnnotationBytes(Executable ex);

    @Positive
    public Class<?>[] getExecutableSharedParameterTypes(Executable ex);

    @Positive
    public Method copyMethod(Method arg);

    @Positive
    public Method leafCopyMethod(Method arg);

    @Positive
    public Field copyField(Field arg);

    @Positive
    public <T> Constructor<T> copyConstructor(Constructor<T> arg);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public <T extends AccessibleObject> T getRoot(T obj);

    @Positive
    public boolean isTrustedFinalField(Field f);

    @Positive
    public <T> T newInstance(Constructor<T> ctor, Object[] args, Class<?> caller) throws IllegalAccessException, InstantiationException, InvocationTargetException;
    @Positive
}

}