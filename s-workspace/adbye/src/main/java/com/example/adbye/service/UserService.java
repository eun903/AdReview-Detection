package com.example.adbye.service;

import java.util.List;
import java.util.Optional;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import com.example.adbye.entity.User;
import com.example.adbye.repository.UserRepository;
import com.example.adbye.dto.AuthRequest;

@Service
public class UserService {

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;
    private final JwtTokenProvider jwtTokenProvider;

    public UserService(UserRepository userRepository, PasswordEncoder passwordEncoder, JwtTokenProvider jwtTokenProvider) {
        this.userRepository = userRepository;
        this.passwordEncoder = passwordEncoder;
        this.jwtTokenProvider = jwtTokenProvider;
    }

//    // 회원가입
//    public void registerUser(String username, String password) {
//        Optional<User> existingUser = userRepository.findByUsername(username);
//        if (existingUser.isPresent()) {
//            throw new IllegalArgumentException("이미 존재하는 아이디입니다.");
//        }
//
//        User newUser = new User();
//        newUser.setUsername(username);
//        newUser.setPassword(passwordEncoder.encode(password)); // 비밀번호 암호화
//        newUser.setRole("ROLE_USER"); // 기본 사용자 역할 부여
//        newUser.setEnabled(true);
//        userRepository.save(newUser);
//    }
    
    // 회원가입 - 현서 코드로 교체
    public void registerUser(AuthRequest authRequest) { // 1. 파라미터를 AuthRequest 객체로 변경
        Optional<User> existingUser = userRepository.findByUsername(authRequest.getUsername());
        if (existingUser.isPresent()) {
            throw new IllegalArgumentException("이미 존재하는 아이디입니다.");
        }

        // 이메일 중복 확인 로직
        if (userRepository.existsByEmail(authRequest.getEmail())) {
            throw new IllegalArgumentException("이미 사용 중인 이메일입니다.");
        }

        User newUser = new User();
        newUser.setUsername(authRequest.getUsername());
        newUser.setPassword(passwordEncoder.encode(authRequest.getPassword()));
        newUser.setEmail(authRequest.getEmail()); // 2. DTO에서 email 값을 가져와 설정
        newUser.setRole("ROLE_USER");
        newUser.setEnabled(true);
        userRepository.save(newUser);
    }

    // 로그인
    public String loginUser(String username, String password) {
        User user = userRepository.findByUsername(username)
            .orElseThrow(() -> new IllegalArgumentException("아이디 또는 비밀번호가 잘못되었습니다."));

        if (!passwordEncoder.matches(password, user.getPassword())) {
            throw new IllegalArgumentException("아이디 또는 비밀번호가 잘못되었습니다.");
        }
        
        if (!user.getEnabled()) {
        	throw new IllegalArgumentException("비활성화된 계정입니다. 관리자에게 문의하세요.");
        }

        return jwtTokenProvider.createToken(user.getUsername(), user.getRole());
    }
    
    public String getUserRole(String username) {
        return userRepository.findByUsername(username)
                             .map(User::getRole)
                             .orElse("ROLE_USER");
    }
    
    // 모든 사용자 조회
    public List<User> findAllUsers() {
        return userRepository.findAll();
    }

    // 사용자 삭제
    public void deleteUserById(Long id) {
        userRepository.deleteById(id);
    }
    
}